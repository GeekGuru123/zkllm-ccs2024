import os
import time

# Define the parameters
model_size = 7  # 7 or 13 (billions of parameters)
sequence_length = 2048  # The sequence length to prove

# Define the input and output file names
input_file = 'layer_input.bin'
attn_input_file = 'attn_input.bin'
attn_output_file = 'attn_output.bin'
post_attn_norm_input_file = 'post_attn_norm_input.bin'
ffn_input_file = 'ffn_input.bin'
ffn_output_file = 'ffn_output.bin'
output_file = 'layer_output.bin'

# Start timing
start_time = time.time()

# Loop through layer numbers from 0 to 31
for layer_number in range(32):
    print(f'Processing layer {layer_number}...')

    # Run the llama-rmsnorm.py script for the first RMSNorm layer
    os.system(f'python llama-rmsnorm.py {model_size} {layer_number} input {sequence_length} --input_file {input_file} --output_file {attn_input_file}')
    
    # Run the llama-self-attn.py script for the self-attention layer
    os.system(f'python llama-self-attn.py {model_size} {layer_number} {sequence_length} --input_file {attn_input_file} --output_file {attn_output_file}')
    
    # Run the llama-skip-connection.py script for the skip connection after the self-attention layer
    os.system(f'python llama-skip-connection.py --block_input_file {input_file} --block_output_file {attn_output_file} --output_file {post_attn_norm_input_file}')
    
    # Run the llama-rmsnorm.py script for the post-attention RMSNorm layer
    os.system(f'python llama-rmsnorm.py {model_size} {layer_number} post_attention {sequence_length} --input_file {post_attn_norm_input_file} --output_file {ffn_input_file}')
    
    # Run the llama-ffn.py script for the feedforward network layer
    os.system(f'python llama-ffn.py {model_size} {layer_number} {sequence_length} --input_file {ffn_input_file} --output_file {ffn_output_file}')
    
    # Run the llama-skip-connection.py script for the skip connection after the feedforward network layer
    os.system(f'python llama-skip-connection.py --block_input_file {post_attn_norm_input_file} --block_output_file {ffn_output_file} --output_file {output_file}')

    # Update input_file for the next layer
    input_file = output_file

# End timing
end_time = time.time()

# Calculate and print the total time taken
total_time = end_time - start_time
print(f'Total time taken: {total_time:.2f} seconds')
