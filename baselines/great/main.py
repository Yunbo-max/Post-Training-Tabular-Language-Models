import pandas as pd
import logging
import sys
import os
import argparse

from baselines.great.models.great import GReaT

def main(args):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    dataname = args.dataname
    batch_size = args.bs
    dataset_path = f'data/{dataname}/train.csv'
    train_df = pd.read_csv(dataset_path)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'


    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # great = GReaT("distilgpt2",                         
    #             epochs=100,                             
    #             save_steps=2000,                     
    #             logging_steps=50,  
    #             fp16=True,                 
    #             experiment_dir=f"{curr_dir}/ckpt/{dataname}",
    #             batch_size=batch_size,
    #             )

    # great = GReaT('gpt2',                         
    #             epochs=100,                             
    #             save_steps=2000,                     
    #             logging_steps=50,  
    #             fp16=True,                 
    #             experiment_dir=f"{curr_dir}/ckpt/{dataname}",
    #             batch_size=batch_size,
    #             )
    
    logger.info("Starting training...")
    trainer = great.fit(train_df)
    logger.info("Training completed successfully")
    great.save(ckpt_dir)
    logger.info(f"Model saved to {ckpt_dir}")

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GReaT')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--bs', type=int, default=16, help='(Maximum) batch size')
    args = parser.parse_args()