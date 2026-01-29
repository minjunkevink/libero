"""
Scene Inventory Extractor for LIBERO datasets.

Extracts objects, fixtures, and affordance regions from BDDL content
stored in HDF5 dataset files.
"""

import h5py
import re
import os
import sys
import tempfile
from typing import Dict, List, Any

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def extract_from_bddl_string(bddl_content: str) -> Dict[str, Any]:
    """
    Parse BDDL string directly to extract scene inventory.
    
    Args:
        bddl_content: BDDL file content as string
        
    Returns:
        Dictionary with objects, fixtures, regions, and language instruction
    """
    inventory = {
        'objects': {},      # {type: [instances]}
        'fixtures': {},     # {type: [instances]}
        'regions': [],      # List of region names
        'language': '',     # Original language instruction
        'obj_of_interest': []  # Objects of interest for the task
    }
    
    # Extract language instruction
    lang_match = re.search(r'\(:language\s+([^)]+)\)', bddl_content)
    if lang_match:
        inventory['language'] = lang_match.group(1).strip()
    
    # Extract objects section: (:objects akita_black_bowl_1 - akita_black_bowl ...)
    obj_match = re.search(r'\(:objects\s+(.*?)\)', bddl_content, re.DOTALL)
    if obj_match:
        obj_content = obj_match.group(1).strip()
        # Parse lines like "akita_black_bowl_1 - akita_black_bowl"
        # or "moka_pot_1 moka_pot_2 - moka_pot"
        current_instances = []
        for token in obj_content.split():
            if token == '-':
                continue
            elif token.endswith('_1') or token.endswith('_2') or token.endswith('_3'):
                # This is an instance name
                current_instances.append(token)
            else:
                # This is a type name, save the instances
                if current_instances:
                    if token not in inventory['objects']:
                        inventory['objects'][token] = []
                    inventory['objects'][token].extend(current_instances)
                    current_instances = []
    
    # Extract fixtures section
    fix_match = re.search(r'\(:fixtures\s+(.*?)\)', bddl_content, re.DOTALL)
    if fix_match:
        fix_content = fix_match.group(1).strip()
        current_instances = []
        for token in fix_content.split():
            if token == '-':
                continue
            elif '_1' in token or '_2' in token or token in ['main_table', 'kitchen_table', 'living_room_table', 'study_table']:
                current_instances.append(token)
            else:
                if current_instances:
                    if token not in inventory['fixtures']:
                        inventory['fixtures'][token] = []
                    inventory['fixtures'][token].extend(current_instances)
                    current_instances = []
    
    # Extract regions
    regions_match = re.search(r'\(:regions\s+(.*?)\)\s*\(:fixtures', bddl_content, re.DOTALL)
    if regions_match:
        regions_content = regions_match.group(1)
        # Find region names (lines starting with '(region_name')
        region_names = re.findall(r'\((\w+_region|\w+_side)\s*$', regions_content, re.MULTILINE)
        inventory['regions'] = region_names
    
    # Extract objects of interest
    ooi_match = re.search(r'\(:obj_of_interest\s+(.*?)\)', bddl_content, re.DOTALL)
    if ooi_match:
        ooi_content = ooi_match.group(1).strip()
        inventory['obj_of_interest'] = ooi_content.split()
    
    return inventory


def extract_scene_inventory(hdf5_path: str) -> Dict[str, Any]:
    """
    Extract scene inventory from BDDL content stored in HDF5 file.
    
    Args:
        hdf5_path: Path to the LIBERO HDF5 dataset file
        
    Returns:
        Dictionary containing:
        - objects: {type: [instances]}
        - fixtures: {type: [instances]}
        - regions: list of region names
        - language: original language instruction
        - all_interactable: flat list of all object/fixture types
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Try to get BDDL content from attributes first
        bddl_content = f['data'].attrs.get('bddl_file_content', '')
        
        if isinstance(bddl_content, bytes):
            bddl_content = bddl_content.decode('utf-8')
        
        # If not found, try to read from bddl_file_name path
        if not bddl_content:
            bddl_file_name = f['data'].attrs.get('bddl_file_name', '')
            if isinstance(bddl_file_name, bytes):
                bddl_file_name = bddl_file_name.decode('utf-8')
            
            if bddl_file_name:
                # Try multiple possible base paths
                possible_paths = [
                    bddl_file_name,  # Absolute or relative as-is
                    os.path.join(os.path.dirname(hdf5_path), '..', '..', bddl_file_name),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(hdf5_path))), bddl_file_name),
                ]
                
                # Also try with libero/ prefix variations
                if not bddl_file_name.startswith('/'):
                    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(hdf5_path)))
                    possible_paths.extend([
                        os.path.join(base_dir, bddl_file_name),
                        os.path.join(base_dir, 'libero', bddl_file_name.replace('libero/', '')),
                        os.path.join(base_dir, bddl_file_name.replace('libero/libero/', 'libero/')),
                    ])
                
                for path in possible_paths:
                    if os.path.exists(path):
                        with open(path, 'r') as bddl_f:
                            bddl_content = bddl_f.read()
                        break
        
        # Fallback: try to get language from problem_info
        if not bddl_content:
            problem_info_str = f['data'].attrs.get('problem_info', '{}')
            if isinstance(problem_info_str, bytes):
                problem_info_str = problem_info_str.decode('utf-8')
            
            import json
            problem_info = json.loads(problem_info_str)
            language = problem_info.get('language_instruction', '')
            
            # Create minimal inventory from problem_info
            # For LIBERO-Goal, we know the scene inventory
            inventory = {
                'objects': {
                    'akita_black_bowl': ['akita_black_bowl_1'],
                    'cream_cheese': ['cream_cheese_1'],
                    'wine_bottle': ['wine_bottle_1'],
                    'plate': ['plate_1']
                },
                'fixtures': {
                    'wooden_cabinet': ['wooden_cabinet_1'],
                    'flat_stove': ['flat_stove_1'],
                    'wine_rack': ['wine_rack_1']
                },
                'regions': ['top_region', 'middle_region', 'bottom_region', 'cook_region'],
                'language': language,
                'obj_of_interest': []
            }
            
            inventory['all_interactable'] = (
                list(inventory['objects'].keys()) + 
                list(inventory['fixtures'].keys())
            )
            return inventory
        
        inventory = extract_from_bddl_string(bddl_content)
        
        # Add convenience field
        inventory['all_interactable'] = (
            list(inventory['objects'].keys()) + 
            list(inventory['fixtures'].keys())
        )
        
        return inventory


def get_human_readable_objects(inventory: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Convert inventory to human-readable object names for LLM prompting.
    
    Args:
        inventory: Scene inventory from extract_scene_inventory()
        
    Returns:
        Dictionary with 'objects' and 'fixtures' as human-readable lists
    """
    def to_readable(name: str) -> str:
        """Convert akita_black_bowl to 'black bowl'"""
        # Remove common prefixes and convert underscores to spaces
        name = name.replace('akita_', '').replace('chefmate_8_', '')
        name = name.replace('_', ' ')
        return name
    
    return {
        'objects': [to_readable(obj) for obj in inventory['objects'].keys()],
        'fixtures': [to_readable(fix) for fix in inventory['fixtures'].keys()],
        'all': [to_readable(item) for item in inventory['all_interactable']]
    }


if __name__ == "__main__":
    # Test with a sample BDDL file
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Extract scene inventory from LIBERO HDF5")
    parser.add_argument("--hdf5", type=str, help="Path to HDF5 file")
    parser.add_argument("--bddl", type=str, help="Path to BDDL file (alternative)")
    args = parser.parse_args()
    
    if args.hdf5:
        inventory = extract_scene_inventory(args.hdf5)
        print("Scene Inventory from HDF5:")
        print(json.dumps(inventory, indent=2))
    elif args.bddl:
        with open(args.bddl, 'r') as f:
            bddl_content = f.read()
        inventory = extract_from_bddl_string(bddl_content)
        # Add all_interactable field for consistency
        inventory['all_interactable'] = (
            list(inventory['objects'].keys()) + 
            list(inventory['fixtures'].keys())
        )
        print("Scene Inventory from BDDL:")
        print(json.dumps(inventory, indent=2))
        print("\nHuman Readable:")
        print(json.dumps(get_human_readable_objects(inventory), indent=2))
    else:
        print("Please provide --hdf5 or --bddl argument")
