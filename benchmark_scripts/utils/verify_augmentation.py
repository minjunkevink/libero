#!/usr/bin/env python3
"""
Verification script for LIBERO language augmentation.

Samples and displays original vs augmented instruction pairs
for manual review, and verifies data integrity.
"""

import h5py
import json
import random
import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def verify_augmentation(
    augmented_hdf5: str,
    original_hdf5: Optional[str] = None,
    num_samples: int = 10,
    check_data_integrity: bool = True
) -> dict:
    """
    Verify augmented dataset integrity and print samples for manual review.
    
    Args:
        augmented_hdf5: Path to augmented HDF5 file
        original_hdf5: Path to original HDF5 file (optional, extracted from metadata)
        num_samples: Number of samples to display
        check_data_integrity: Whether to verify ExternalLink data access
        
    Returns:
        Dictionary with verification results
    """
    results = {
        'augmented_file': augmented_hdf5,
        'original_file': original_hdf5,
        'total_demos': 0,
        'samples_checked': 0,
        'data_integrity_ok': True,
        'errors': []
    }
    
    print(f"\n{'='*60}")
    print("AUGMENTATION VERIFICATION")
    print(f"{'='*60}")
    print(f"Augmented file: {augmented_hdf5}")
    
    with h5py.File(augmented_hdf5, 'r') as f_aug:
        # Get metadata
        if 'augmentation_info' in f_aug['data'].attrs:
            aug_info = json.loads(f_aug['data'].attrs['augmentation_info'])
            original_hdf5 = original_hdf5 or aug_info.get('original_file', '')
            print(f"Original file: {original_hdf5}")
            print(f"Original instruction: {aug_info.get('original_instruction', 'N/A')}")
            print(f"Variations per demo: {aug_info.get('variations_per_demo', 'N/A')}")
            print(f"Objects: {aug_info.get('objects', [])}")
            print(f"Fixtures: {aug_info.get('fixtures', [])}")
        
        results['original_file'] = original_hdf5
        
        # Get all demos
        demos = [k for k in f_aug['data'].keys() if k.startswith('demo')]
        demos = sorted(demos, key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        results['total_demos'] = len(demos)
        
        print(f"\nTotal augmented demos: {len(demos)}")
        
        # Sample demos for review
        sample_demos = random.sample(demos, min(num_samples, len(demos)))
        results['samples_checked'] = len(sample_demos)
        
        print(f"\n{'-'*60}")
        print(f"SAMPLE REVIEW ({len(sample_demos)} samples)")
        print(f"{'-'*60}")
        
        for demo in sample_demos:
            demo_grp = f_aug[f'data/{demo}']
            
            # Get problem info
            problem_info_str = demo_grp.attrs.get('problem_info', '{}')
            if isinstance(problem_info_str, bytes):
                problem_info_str = problem_info_str.decode('utf-8')
            
            try:
                problem_info = json.loads(problem_info_str)
            except json.JSONDecodeError:
                problem_info = {}
            
            original = problem_info.get('original_instruction', 'N/A')
            augmented = problem_info.get('language_instruction', 'N/A')
            is_augmented = problem_info.get('is_augmented', False)
            source_demo = problem_info.get('source_demo', 'N/A')
            aug_idx = problem_info.get('augmentation_index', 'N/A')
            
            print(f"\n[{demo}] (source: {source_demo}, idx: {aug_idx})")
            print(f"  Original:  {original}")
            print(f"  Augmented: {augmented}")
            print(f"  Is augmented: {is_augmented}")
            
            # Check data integrity if requested
            if check_data_integrity:
                try:
                    # Try to access observation data through ExternalLink
                    if 'obs' in demo_grp:
                        obs_grp = demo_grp['obs']
                        # Check if we can read a sample
                        if 'agentview_rgb' in obs_grp:
                            shape = obs_grp['agentview_rgb'].shape
                            print(f"  Data OK: agentview_rgb {shape}")
                        elif len(list(obs_grp.keys())) > 0:
                            first_key = list(obs_grp.keys())[0]
                            shape = obs_grp[first_key].shape
                            print(f"  Data OK: {first_key} {shape}")
                        else:
                            print(f"  Data OK: obs group accessible")
                    
                    if 'actions' in demo_grp:
                        actions_shape = demo_grp['actions'].shape
                        print(f"  Data OK: actions {actions_shape}")
                        
                except Exception as e:
                    error_msg = f"Data access error in {demo}: {e}"
                    print(f"  ERROR: {error_msg}")
                    results['errors'].append(error_msg)
                    results['data_integrity_ok'] = False
        
        # Summary
        print(f"\n{'-'*60}")
        print("VERIFICATION SUMMARY")
        print(f"{'-'*60}")
        print(f"Total demos: {results['total_demos']}")
        print(f"Samples checked: {results['samples_checked']}")
        print(f"Data integrity: {'OK' if results['data_integrity_ok'] else 'FAILED'}")
        
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for err in results['errors']:
                print(f"  - {err}")
        
        print(f"{'='*60}\n")
    
    return results


def compare_original_augmented(augmented_hdf5: str, original_hdf5: str):
    """
    Compare original and augmented datasets side by side.
    """
    print(f"\n{'='*60}")
    print("ORIGINAL vs AUGMENTED COMPARISON")
    print(f"{'='*60}")
    
    with h5py.File(original_hdf5, 'r') as f_orig:
        with h5py.File(augmented_hdf5, 'r') as f_aug:
            orig_demos = [k for k in f_orig['data'].keys() if k.startswith('demo')]
            aug_demos = [k for k in f_aug['data'].keys() if k.startswith('demo')]
            
            print(f"Original demos: {len(orig_demos)}")
            print(f"Augmented demos: {len(aug_demos)}")
            print(f"Expansion: {len(aug_demos) / len(orig_demos):.1f}x")
            
            # Get original instruction
            orig_info_str = f_orig['data'].attrs.get('problem_info', '{}')
            if isinstance(orig_info_str, bytes):
                orig_info_str = orig_info_str.decode('utf-8')
            orig_info = json.loads(orig_info_str)
            orig_instruction = orig_info.get('language_instruction', 'N/A')
            
            print(f"\nOriginal instruction: {orig_instruction}")
            
            # Show unique augmented instructions
            unique_instructions = set()
            for demo in aug_demos[:100]:  # Check first 100
                demo_grp = f_aug[f'data/{demo}']
                info_str = demo_grp.attrs.get('problem_info', '{}')
                if isinstance(info_str, bytes):
                    info_str = info_str.decode('utf-8')
                info = json.loads(info_str)
                instr = info.get('language_instruction', '')
                if instr and instr != orig_instruction:
                    unique_instructions.add(instr)
            
            print(f"\nUnique augmented instructions ({len(unique_instructions)}):")
            for i, instr in enumerate(sorted(unique_instructions)[:15], 1):
                print(f"  {i}. {instr}")
            
            if len(unique_instructions) > 15:
                print(f"  ... and {len(unique_instructions) - 15} more")


def main():
    parser = argparse.ArgumentParser(
        description="Verify LIBERO language augmentation results"
    )
    parser.add_argument(
        "--augmented",
        type=str,
        required=True,
        help="Path to augmented HDF5 file"
    )
    parser.add_argument(
        "--original",
        type=str,
        help="Path to original HDF5 file (optional)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to display"
    )
    parser.add_argument(
        "--no_data_check",
        action="store_true",
        help="Skip data integrity check"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare original vs augmented stats"
    )
    
    args = parser.parse_args()
    
    # Basic verification
    results = verify_augmentation(
        args.augmented,
        args.original,
        args.num_samples,
        check_data_integrity=not args.no_data_check
    )
    
    # Optional comparison
    if args.compare and args.original:
        compare_original_augmented(args.augmented, args.original)
    
    # Exit code based on results
    if results['data_integrity_ok']:
        print("Verification PASSED")
        sys.exit(0)
    else:
        print("Verification FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
