#!/usr/bin/env python3
"""Scan R2 bucket for images and link them to memories in D1.

This script:
1. Lists all images in the R2 bucket under images/{memory_id}/
2. Groups them by memory_id
3. Updates each memory's metadata in D1 to include image references

Usage:
    python link-r2-images.py --bucket ob1 --d1-id <database-id> [--dry-run]

Examples:
    AWS_PROFILE=myprofile python link-r2-images.py --bucket my-bucket --d1-id <database-id>
"""

import argparse
import json
import os
import re
from collections import defaultdict

import boto3
import requests
from botocore.config import Config

# Cloudflare account ID (from environment or default)
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")


def get_r2_client():
    """Get boto3 S3 client configured for R2."""
    endpoint_url = os.getenv(
        "AWS_ENDPOINT_URL",
        f"https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com"
    )

    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        config=Config(signature_version='s3v4')
    )


def list_r2_images(s3_client, bucket: str) -> dict[int, list[dict]]:
    """List all images in R2 and group by memory_id."""
    images_by_memory = defaultdict(list)

    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix='images/'):
        for obj in page.get('Contents', []):
            key = obj['Key']

            # Skip directory markers
            if key.endswith('/') or obj['Size'] == 0:
                continue

            # Parse key: images/{memory_id}/{timestamp}_{index}_{hash}.{ext}
            match = re.match(r'images/(\d+)/(\d+)_(\d+)_([a-f0-9]+)\.(\w+)', key)
            if match:
                memory_id = int(match.group(1))
                timestamp = int(match.group(2))
                index = int(match.group(3))
                ext = match.group(5)

                images_by_memory[memory_id].append({
                    'key': key,
                    'timestamp': timestamp,
                    'index': index,
                    'ext': ext,
                    'size': obj['Size'],
                    'src': f'r2://{key}',
                })

    # Sort images by index within each memory
    for memory_id in images_by_memory:
        images_by_memory[memory_id].sort(key=lambda x: (x['timestamp'], x['index']))

    return dict(images_by_memory)


def d1_query(d1_id: str, api_token: str, sql: str) -> list[dict] | None:
    """Execute a D1 query via Cloudflare API."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{d1_id}/query"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        json={"sql": sql}
    )

    if response.status_code != 200:
        print(f"  API error: {response.status_code} - {response.text}")
        return None

    data = response.json()
    if not data.get('success'):
        print(f"  D1 error: {data.get('errors')}")
        return None

    return data['result'][0].get('results', [])


def get_memory_metadata(d1_id: str, api_token: str, memory_id: int) -> dict | None:
    """Get current metadata for a memory from D1."""
    results = d1_query(d1_id, api_token, f"SELECT metadata FROM memories WHERE id = {memory_id}")

    if results is None:
        return None

    if not results:
        return None  # Memory not found

    meta_str = results[0].get('metadata')
    return json.loads(meta_str) if meta_str else {}


def update_memory_metadata(d1_id: str, api_token: str, memory_id: int, metadata: dict, dry_run: bool) -> bool:
    """Update memory metadata in D1."""
    if dry_run:
        print(f"  [DRY RUN] Would update memory {memory_id}")
        return True

    meta_json = json.dumps(metadata).replace("'", "''")  # Escape single quotes for SQL
    sql = f"UPDATE memories SET metadata = '{meta_json}' WHERE id = {memory_id}"

    results = d1_query(d1_id, api_token, sql)
    return results is not None


def main():
    parser = argparse.ArgumentParser(description='Link R2 images to D1 memories')
    parser.add_argument('--bucket', required=True, help='R2 bucket name (e.g., ob1, memora)')
    parser.add_argument('--d1-id', required=True, help='D1 database ID')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    args = parser.parse_args()

    # Get API token and account ID from environment
    api_token = os.getenv('CLOUDFLARE_API_TOKEN') or os.getenv('CF_API_TOKEN')
    if not api_token:
        print("Error: CLOUDFLARE_API_TOKEN or CF_API_TOKEN environment variable required")
        return 1

    if not CF_ACCOUNT_ID:
        print("Error: CF_ACCOUNT_ID environment variable required")
        return 1

    print(f"Scanning R2 bucket '{args.bucket}' for images...")

    s3_client = get_r2_client()
    images_by_memory = list_r2_images(s3_client, args.bucket)

    if not images_by_memory:
        print("No images found in R2 bucket.")
        return 0

    print(f"Found images for {len(images_by_memory)} memories:")
    for memory_id, images in sorted(images_by_memory.items()):
        print(f"  Memory {memory_id}: {len(images)} images")

    print(f"\nLinking images to D1 database '{args.d1_id}'...")

    updated = 0
    skipped = 0
    errors = 0

    for memory_id, images in sorted(images_by_memory.items()):
        # Get current metadata
        metadata = get_memory_metadata(args.d1_id, api_token, memory_id)

        if metadata is None:
            print(f"  Memory {memory_id}: NOT FOUND in D1, skipping")
            skipped += 1
            continue

        # Check if images already linked
        existing_images = metadata.get('images', [])
        existing_srcs = {img.get('src') for img in existing_images}

        # Build new images list
        new_images = []
        for img in images:
            if img['src'] not in existing_srcs:
                new_images.append({
                    'src': img['src'],
                    'caption': '',
                })

        if not new_images:
            print(f"  Memory {memory_id}: already has all {len(images)} images linked")
            skipped += 1
            continue

        # Update metadata with new images
        metadata['images'] = existing_images + new_images

        print(f"  Memory {memory_id}: linking {len(new_images)} new images")

        if update_memory_metadata(args.d1_id, api_token, memory_id, metadata, args.dry_run):
            updated += 1
        else:
            errors += 1

    print(f"\nDone! Updated: {updated}, Skipped: {skipped}, Errors: {errors}")

    if args.dry_run:
        print("\n(This was a dry run - no changes were made)")

    return 0


if __name__ == '__main__':
    exit(main())
