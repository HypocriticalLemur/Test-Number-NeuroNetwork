from imports import *

IMAGE_SIDE_SIZE = 28
def Save(bias_input_to_hidden, weights_input_to_hidden, bias_hidden_to_output, weights_hidden_to_output, fileName = "legacy_network"):
	np.savez(fileName+".npz", bias_input_to_hidden, weights_input_to_hidden, bias_hidden_to_output, weights_hidden_to_output, encoding = "bytes")
def Load(fileName = "legacy_network"):
	pack = np.load(fileName + ".npz",encoding='bytes')
	unpack = pack["arr_0"],pack["arr_1"],pack["arr_2"],pack["arr_3"]
	return unpack
	
def Check(bias_input_to_hidden, weights_input_to_hidden, bias_hidden_to_output, weights_hidden_to_output, imgName = "custom.jpg"):
	# CHECK CUSTOM
	test_image = plt.imread(imgName, format="jpeg")

	# Grayscale + Unit RGB + inverse colors
	gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
	test_image = 1 - (gray(test_image).astype("float32") / 255)

	# Reshape
	test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

	# Predict
	image = np.reshape(test_image, (-1, 1))

	# Forward propagation (to hidden layer)
	hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
	hidden = Sigmoid(hidden_raw) # sigmoid
	# Forward propagation (to output layer)
	output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
	output = Sigmoid(output_raw)
	guess = output.argmax()

	plt.imshow(test_image.reshape(IMAGE_SIDE_SIZE, IMAGE_SIDE_SIZE), cmap="Greys")
	plt.title(f"NN suggests the CUSTOM number is: {guess}")
	plt.show()
	return guess

def Run():
	images, labels = utils.load_dataset()
	inpuNeuronsNum = images.shape[1]
	outputNeuronsNum = labels.shape[1]
	hiddenNuronsNum = outputNeuronsNum*2
	weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (hiddenNuronsNum, inpuNeuronsNum))
	weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (outputNeuronsNum, hiddenNuronsNum))
	bias_input_to_hidden = np.zeros((hiddenNuronsNum, 1))
	bias_hidden_to_output = np.zeros((outputNeuronsNum, 1))

	epochs = 4
	e_loss = 0
	e_correct = 0
	learning_rate = 0.01

	for epoch in range(epochs):
		print(f"Epoch №{epoch}")

		for i, (image, label) in enumerate(zip(images, labels)):
			sys.stdout.write(f"{round(float(i)/labels.shape[0]*100)}%\r")
			image = np.reshape(image, (-1, 1))
			label = np.reshape(label, (-1, 1))

			# print("Forward... 1/2", end='')
			# Forward propagation (to hidden layer)
			hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
			hidden = Sigmoid(hidden_raw) # sigmoid
			# print(" 2/2")
			# Forward propagation (to output layer)
			output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
			output = Sigmoid(output_raw)

			# print("Calculating error...")
			# Loss / Error calculation
			e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
			e_correct += int(np.argmax(output) == np.argmax(label))
			
			# print("Back... 1/2", end='')
			# Backpropagation (output layer)
			delta_output = output - label
			weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
			bias_hidden_to_output += -learning_rate * delta_output

			# print(" 2/2")
			# Backpropagation (hidden layer)
			tmp = (hidden * (1 - hidden))
			delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * tmp
			weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
			bias_input_to_hidden += -learning_rate * delta_hidden

			# DONE

		# print some debug info between epochs
		print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
		print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")
		e_loss = 0
		e_correct = 0
	Save(bias_input_to_hidden, weights_input_to_hidden, bias_hidden_to_output, weights_hidden_to_output)
	Check(bias_input_to_hidden, weights_input_to_hidden, bias_hidden_to_output, weights_hidden_to_output)
