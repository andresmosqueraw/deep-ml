import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	flatten_kernel = [v for row in kernel for v in row]

	# Your code here
	padded_input = np.pad(input_matrix, padding, mode='constant')
	input_height_pad, input_width_pad = padded_input.shape

	output_height = ((input_height_pad - kernel_height) // stride) + 1
	output_width = ((input_width_pad - kernel_width) // stride) + 1
	# output_len = ((input_height - kernel_height) + (2*padding) // stride) + 1

	output = np.zeros((output_height, output_width))

	for i in range(output_height):
		for j in range(output_width):
			init_row = i * stride
			end_row = (i * stride) + kernel_height
			init_col = j * stride
			end_col = (j * stride) + kernel_width
			# region = [v for fila in padded_input[row_init:row_end] for v in fila]
			region = [valor for fila in padded_input[init_row:end_row] for valor in fila[init_col:end_col]]
			# print(flatten_kernel)
			# print(region)
			output[i][j] = sum(flatten_kernel[i] * region[i] for i in range(len(region)))

	return output
