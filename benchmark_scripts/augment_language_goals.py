#!/usr/bin/env python3
"""
Scene-Aware Language Augmentation for LIBERO-Goal

This script generates augmented HDF5 datasets where the visual trajectory
is unchanged, but the language instruction is extended with a "phantom task"
(e.g., "Pick up the bowl" -> "Pick up the bowl and then put it on the stove").

Key Features:
- Uses HDF5 ExternalLinks to avoid duplicating heavy image data
- Scene-grounded generation: only references objects that exist in the scene
- Caches LLM responses to avoid redundant API calls
- Supports mock mode for testing without API costs

Usage:
    python augment_language_goals.py --input_dir /path/to/libero_goal --output_dir /path/to/augmented
"""

import init_path
import argparse
import h5py
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

from utils.extract_scene_inventory import extract_scene_inventory, get_human_readable_objects
from utils.llm_augmenter import LLMLanguageAugmenter


# LIBERO-Goal task list for validation
LIBERO_GOAL_TASKS = [
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_top_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_cream_cheese_in_the_bowl",
    "turn_on_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
]


def validate_libero_goal(hdf5_path: str) -> bool:
    """
    Validate that the HDF5 file is a LIBERO-Goal dataset.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        True if valid LIBERO-Goal dataset
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Check for required attributes
            if 'data' not in f:
                return False
            
            # Check problem info
            problem_info = f['data'].attrs.get('problem_info', '')
            if isinstance(problem_info, bytes):
                problem_info = problem_info.decode('utf-8')
            
            if problem_info:
                info = json.loads(problem_info)
                # LIBERO-Goal uses "LIBERO_Tabletop_Manipulation" problem
                problem_name = info.get('problem_name', '')
                if 'Tabletop_Manipulation' in problem_name:
                    return True
            
            # Also check bddl content
            bddl_content = f['data'].attrs.get('bddl_file_content', '')
            if isinstance(bddl_content, bytes):
                bddl_content = bddl_content.decode('utf-8')
            
            if 'LIBERO_Tabletop_Manipulation' in bddl_content:
                return True
                
        return False
    except Exception as e:
        print(f"Warning: Could not validate {hdf5_path}: {e}")
        return False


def augment_single_dataset(
    input_hdf5: str,
    output_hdf5: str,
    augmenter: LLMLanguageAugmenter,
    num_variations: int = 10,
    include_original: bool = True
) -> Dict[str, Any]:
    """
    Create augmented dataset with extended language instructions.
    
    Args:
        input_hdf5: Path to original HDF5 file
        output_hdf5: Path for augmented output file
        augmenter: LLM augmenter instance
        num_variations: Number of augmented variations per demo
        include_original: Whether to include original instruction as first entry
        
    Returns:
        Statistics about the augmentation
    """
    stats = {
        'input_file': input_hdf5,
        'output_file': output_hdf5,
        'original_demos': 0,
        'augmented_demos': 0,
        'variations_per_demo': num_variations,
        'original_instruction': '',
        'sample_augmentations': []
    }
    
    # Validate input
    if not validate_libero_goal(input_hdf5):
        print(f"Warning: {input_hdf5} may not be a LIBERO-Goal dataset")
    
    with h5py.File(input_hdf5, 'r') as f_src:
        # Extract scene inventory
        inventory = extract_scene_inventory(input_hdf5)
        readable = get_human_readable_objects(inventory)
        
        objects = readable['objects']
        fixtures = readable['fixtures']
        
        # Get original metadata
        problem_info_str = f_src['data'].attrs.get('problem_info', '{}')
        if isinstance(problem_info_str, bytes):
            problem_info_str = problem_info_str.decode('utf-8')
        problem_info = json.loads(problem_info_str)
        
        original_instruction = problem_info.get('language_instruction', '')
        if not original_instruction:
            original_instruction = inventory.get('language', '')
        
        stats['original_instruction'] = original_instruction
        
        print(f"\n{'='*60}")
        print(f"Processing: {Path(input_hdf5).name}")
        print(f"Original instruction: {original_instruction}")
        print(f"Scene objects: {objects}")
        print(f"Scene fixtures: {fixtures}")
        print(f"{'='*60}")
        
        # Generate augmented instructions
        extended_instructions = augmenter.generate_extensions(
            original_instruction, objects, fixtures, num_variations
        )
        
        stats['sample_augmentations'] = extended_instructions[:3]
        
        # Get list of demos
        demos = [k for k in f_src['data'].keys() if k.startswith('demo')]
        demos = sorted(demos, key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        stats['original_demos'] = len(demos)
        
        print(f"Found {len(demos)} original demos")
        print(f"Generating {num_variations} variations each...")
        
        # Create output file
        os.makedirs(os.path.dirname(output_hdf5), exist_ok=True)
        
        with h5py.File(output_hdf5, 'w') as f_dst:
            # Create data group
            data_grp = f_dst.create_group('data')
            
            # Copy top-level attributes (except problem_info which we'll modify)
            for attr_name in f_src['data'].attrs.keys():
                attr_val = f_src['data'].attrs[attr_name]
                if attr_name == 'problem_info':
                    # Will be set per-demo
                    continue
                data_grp.attrs[attr_name] = attr_val
            
            demo_counter = 0
            
            for demo_name in demos:
                src_demo = f_src[f'data/{demo_name}']
                
                # Prepare instruction list (original + augmented)
                if include_original:
                    all_instructions = [original_instruction] + extended_instructions
                else:
                    all_instructions = extended_instructions
                
                # Create augmented entries
                for var_idx, new_instruction in enumerate(all_instructions):
                    new_demo_name = f"demo_{demo_counter}"
                    demo_counter += 1
                    
                    dst_demo = f_dst.create_group(f'data/{new_demo_name}')
                    
                    # Use ExternalLinks for heavy data to avoid duplication
                    # Convert to absolute path for reliability
                    abs_input_path = os.path.abspath(input_hdf5)
                    
                    for key in src_demo.keys():
                        if key in ['obs', 'next_obs']:
                            # For observation groups, link the entire group
                            f_dst[f'data/{new_demo_name}/{key}'] = h5py.ExternalLink(
                                abs_input_path, f'data/{demo_name}/{key}'
                            )
                        elif isinstance(src_demo[key], h5py.Dataset):
                            # Link datasets (actions, states, etc.)
                            f_dst[f'data/{new_demo_name}/{key}'] = h5py.ExternalLink(
                                abs_input_path, f'data/{demo_name}/{key}'
                            )
                        elif isinstance(src_demo[key], h5py.Group):
                            # Link groups
                            f_dst[f'data/{new_demo_name}/{key}'] = h5py.ExternalLink(
                                abs_input_path, f'data/{demo_name}/{key}'
                            )
                    
                    # Copy attributes from source demo
                    for attr_name in src_demo.attrs.keys():
                        dst_demo.attrs[attr_name] = src_demo.attrs[attr_name]
                    
                    # Create augmented problem_info
                    new_problem_info = problem_info.copy()
                    new_problem_info['language_instruction'] = new_instruction
                    new_problem_info['original_instruction'] = original_instruction
                    new_problem_info['is_augmented'] = (var_idx > 0) if include_original else True
                    new_problem_info['augmentation_index'] = var_idx
                    new_problem_info['source_demo'] = demo_name
                    
                    # Store as attribute
                    dst_demo.attrs['problem_info'] = json.dumps(new_problem_info)
            
            # Update metadata
            data_grp.attrs['num_demos'] = demo_counter
            data_grp.attrs['augmentation_info'] = json.dumps({
                'original_file': abs_input_path,
                'original_instruction': original_instruction,
                'variations_per_demo': num_variations,
                'include_original': include_original,
                'total_demos': demo_counter,
                'objects': objects,
                'fixtures': fixtures
            })
            
            stats['augmented_demos'] = demo_counter
    
    print(f"Created {stats['augmented_demos']} augmented demos")
    print(f"Output: {output_hdf5}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Scene-Aware Language Augmentation for LIBERO-Goal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Augment all LIBERO-Goal datasets with mock LLM (testing)
  python augment_language_goals.py --input_dir ./datasets/libero_goal --output_dir ./datasets/libero_goal_augmented --mock_llm

  # Augment with real OpenAI API
  python augment_language_goals.py --input_dir ./datasets/libero_goal --output_dir ./datasets/libero_goal_augmented

  # Augment single file
  python augment_language_goals.py --input_file ./datasets/libero_goal/put_the_bowl_on_the_stove_demo.hdf5 --output_dir ./augmented
        """
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing LIBERO-Goal HDF5 files"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Single HDF5 file to augment (alternative to --input_dir)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for augmented datasets"
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=10,
        help="Number of augmented variations per demo (default: 10)"
    )
    parser.add_argument(
        "--mock_llm",
        action="store_true",
        help="Use mock LLM responses (for testing without API costs)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./augmentation_cache",
        help="Directory to cache LLM responses"
    )
    parser.add_argument(
        "--no_include_original",
        action="store_true",
        help="Don't include original instruction as first variation"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input_dir and not args.input_file:
        parser.error("Either --input_dir or --input_file must be specified")
    
    # Initialize augmenter
    augmenter = LLMLanguageAugmenter(
        cache_dir=args.cache_dir,
        mock=args.mock_llm,
        model=args.model
    )
    
    # Collect input files
    input_files = []
    if args.input_file:
        input_files = [args.input_file]
    else:
        input_dir = Path(args.input_dir)
        input_files = sorted(input_dir.glob("*_demo.hdf5"))
        if not input_files:
            # Also try without _demo suffix
            input_files = sorted(input_dir.glob("*.hdf5"))
    
    if not input_files:
        print(f"Error: No HDF5 files found in {args.input_dir or args.input_file}")
        sys.exit(1)
    
    print(f"\n{'#'*60}")
    print("LIBERO-Goal Language Augmentation")
    print(f"{'#'*60}")
    print(f"Input files: {len(input_files)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Variations per demo: {args.num_variations}")
    print(f"LLM mode: {'Mock' if args.mock_llm else f'OpenAI ({args.model})'}")
    print(f"Include original: {not args.no_include_original}")
    print(f"{'#'*60}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each file
    all_stats = []
    for input_file in input_files:
        input_path = str(input_file)
        
        # Generate output filename
        input_name = Path(input_path).stem
        if input_name.endswith('_demo'):
            output_name = input_name.replace('_demo', '_augmented.hdf5')
        else:
            output_name = f"{input_name}_augmented.hdf5"
        
        output_path = os.path.join(args.output_dir, output_name)
        
        try:
            stats = augment_single_dataset(
                input_path,
                output_path,
                augmenter,
                num_variations=args.num_variations,
                include_original=not args.no_include_original
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("AUGMENTATION SUMMARY")
    print(f"{'='*60}")
    
    total_original = sum(s['original_demos'] for s in all_stats)
    total_augmented = sum(s['augmented_demos'] for s in all_stats)
    
    print(f"Files processed: {len(all_stats)}")
    print(f"Total original demos: {total_original}")
    print(f"Total augmented demos: {total_augmented}")
    print(f"Expansion factor: {total_augmented / total_original:.1f}x" if total_original > 0 else "N/A")
    
    print(f"\nSample augmentations:")
    for stats in all_stats[:3]:
        print(f"\n  Original: {stats['original_instruction']}")
        for aug in stats['sample_augmentations'][:2]:
            print(f"  -> {aug}")
    
    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
