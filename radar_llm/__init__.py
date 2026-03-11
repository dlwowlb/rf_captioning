# RadarLLM: Radar-based Human Activity Captioning
# Baseline reproduction of Motion-guided Radar Tokenizer + Radar-aware Language Model
#
# End-to-end pipeline:
#   Text Prompt → HY-Motion → SMPL → RF-Genesis Radar Simulation
#   → Point Cloud Extraction → VQ-VAE Tokenizer → FLAN-T5-small Captioning
#
# Usage:
#   python -m radar_llm.run_pipeline --mode all          # Full pipeline
#   python -m radar_llm.generate_data                    # Data generation only
#   python -m radar_llm.train_tokenizer --data_root ...  # Stage 1
#   python -m radar_llm.train_lm --data_root ...         # Stage 2
