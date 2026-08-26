import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	pad_input = np.pad(input_matrix, padding, mode='constant')
	pad_height, pad_width = pad_input.shape

	output_height = ((pad_height - kernel_height) // stride) + 1
	output_width = ((pad_width - kernel_width) // stride) + 1
	output_matrix = np.zeros((output_height, output_width))

	for i in range(output_height):
		for j in range(output_width):
			init_row = i * stride
			end_row = (i * stride) + kernel_height
			init_col = j * stride
			end_col = (j * stride) + kernel_width
			region = pad_input[init_row:end_row, init_col:end_col]
			output_matrix[i,j] = np.sum(region * kernel)
    
	return output_matrix
