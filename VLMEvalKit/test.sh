export LMUData=/datasets/VLMEval

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --master_port=29500 --nproc_per_node=8 run.py --data DynaMath WeMath MathVerse_MINI HRBench4K HRBench8K VStarBench MMStar RealWorldQA --model SwimBird-SFT-8B --judge your_judge_model --api-nproc 10 --verbose 
