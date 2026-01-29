# LIBERO Benchmark Scripts

This directory contains scripts for data augmentation and processing of LIBERO datasets.

## Environment Setup

**Important:** The augmentation scripts require a standard CPython environment, NOT the GraalVM-based `libero` conda environment.

| Task | Environment |
|------|-------------|
| Language Augmentation | `base` (or any CPython env) |
| LIBERO Simulations | `libero` (GraalVM) |
| Training/Evaluation | `libero` (GraalVM) |

### Quick Setup

```bash
# For augmentation scripts
conda activate base
pip install h5py python-dotenv openai

# Set your OpenAI API key
echo 'OPENAI_API_KEY="sk-your-key-here"' > .env
```

## Scripts Overview

### `augment_language_goals.py`

**Scene-Aware Language Augmentation for LIBERO-Goal**

This script generates augmented HDF5 datasets where the visual trajectory is unchanged, but the language instruction is extended with a "phantom task" (e.g., "Pick up the bowl" → "Pick up the bowl and then put it on the stove").

#### Features
- Uses HDF5 ExternalLinks to avoid duplicating heavy image data (~95% space savings)
- Scene-grounded generation: only references objects that exist in the scene
- Caches LLM responses to avoid redundant API calls
- Supports mock mode for testing without API costs

#### Usage

```bash
# Activate base environment (NOT libero!)
conda activate base

# Augment ALL LIBERO-Goal tasks with real LLM
python benchmark_scripts/augment_language_goals.py \
    --input_dir libero/datasets/libero_goal \
    --output_dir ./datasets/libero_goal_augmented \
    --num_variations 10

# Test with mock LLM (no API costs)
python benchmark_scripts/augment_language_goals.py \
    --input_dir libero/datasets/libero_goal \
    --output_dir ./datasets/libero_goal_augmented_mock \
    --num_variations 5 \
    --mock_llm

# Augment a single file
python benchmark_scripts/augment_language_goals.py \
    --input_file libero/datasets/libero_goal/put_the_bowl_on_the_stove_demo.hdf5 \
    --output_dir ./augmented \
    --num_variations 10
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_dir` | Directory containing LIBERO-Goal HDF5 files | - |
| `--input_file` | Single HDF5 file to augment | - |
| `--output_dir` | Output directory for augmented datasets | Required |
| `--num_variations` | Number of augmented variations per demo | 10 |
| `--mock_llm` | Use mock LLM responses (for testing) | False |
| `--model` | OpenAI model to use | gpt-4o-mini |
| `--cache_dir` | Directory to cache LLM responses | ./augmentation_cache |
| `--no_include_original` | Don't include original instruction | False |

#### Output

For each input file `task_demo.hdf5`, creates `task_augmented.hdf5` containing:
- Original demos with original instruction (if `--no_include_original` not set)
- N augmented demos per original demo with extended instructions

Example expansion: 50 demos × (1 original + 10 augmented) = 550 augmented demos

### `utils/verify_augmentation.py`

**Verify augmented dataset integrity**

```bash
python benchmark_scripts/utils/verify_augmentation.py \
    --augmented ./datasets/libero_goal_augmented/put_the_bowl_on_the_stove_augmented.hdf5 \
    --num_samples 10

# Compare with original
python benchmark_scripts/utils/verify_augmentation.py \
    --augmented ./augmented/task_augmented.hdf5 \
    --original ./libero/datasets/libero_goal/task_demo.hdf5 \
    --compare
```

### `utils/extract_scene_inventory.py`

**Extract scene objects and fixtures from BDDL**

```bash
# From HDF5 file
python benchmark_scripts/utils/extract_scene_inventory.py \
    --hdf5 libero/datasets/libero_goal/put_the_bowl_on_the_stove_demo.hdf5

# From BDDL file directly
python benchmark_scripts/utils/extract_scene_inventory.py \
    --bddl libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_stove.bddl
```

### `utils/llm_augmenter.py`

**LLM-based language augmentation with caching**

```bash
# Test with real API
python benchmark_scripts/utils/llm_augmenter.py \
    --instruction "put the bowl on the stove" \
    --num 5

# Test with mock responses
python benchmark_scripts/utils/llm_augmenter.py \
    --instruction "put the bowl on the stove" \
    --num 5 \
    --mock
```

## Project Structure

```
benchmark_scripts/
├── README.md                      # This file
├── augment_language_goals.py      # Main augmentation pipeline
├── init_path.py                   # Path initialization
├── prompts/
│   └── augmentation_prompt.txt    # LLM prompt template
└── utils/
    ├── __init__.py
    ├── extract_scene_inventory.py # BDDL parser for scene objects
    ├── llm_augmenter.py           # LLM wrapper with caching
    └── verify_augmentation.py     # Dataset verification
```

## Caching

LLM responses are cached in `./augmentation_cache/` to avoid redundant API calls. Each unique (instruction, objects, fixtures, num_variations) combination gets its own cache file.

To clear the cache:
```bash
rm -rf ./augmentation_cache/
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'h5py'"
Make sure you're using the `base` environment, not `libero`:
```bash
conda activate base
pip install h5py
```

### "OPENAI_API_KEY not found"
Create a `.env` file in the project root:
```bash
echo 'OPENAI_API_KEY="sk-your-key-here"' > .env
```

### GraalVM Python issues
The `libero` conda environment uses GraalVM Python which has compatibility issues with some packages. Use `base` or another CPython environment for augmentation scripts.
