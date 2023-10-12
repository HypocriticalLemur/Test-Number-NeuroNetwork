from imports import *
import legacy

class Neuron:
	"not used"
	def __init__(self, numInput):
		self.weights = np.zeros(numInput)
class NeuroLayer:
	"one layer of neural network"
	def __init__(self, numInputs, numNeurons):
		self.weights = np.random.uniform(-0.5, 0.5, (numNeurons, numInputs))
		self.bias = np.zeros((numNeurons, 1))
		self.neuroValues = np.zeros((numNeurons, 1))	#the values before activation function
	def CalculateOutput(self, input:np.ndarray):
		"calculating values of the current layer"
		# print("hi")
		self.neuroValues = self.bias +  self.weights @ input
		return self.neuroValues

class NeuroWork:
	def __init__(self, numHiddenLayers, numInputNeurons, numOutputNeurons):
		self.learningRate = .01
		self.layers:list[NeuroLayer] = []#[NeuroLayer(numInputNeurons, numOutputNeurons*2)]	#we create at least 1 hidden layer
		prevInputNum = numInputNeurons
		for _ in range(numHiddenLayers):
			tmpN = round(prevInputNum/10)
			self.layers.append(NeuroLayer(prevInputNum, tmpN))
			prevInputNum = tmpN
			
		self.layers.append(NeuroLayer(prevInputNum, numOutputNeurons))
	def Forward(self, input:np.ndarray):
		self.input = input
		if input.shape[0] != self.layers[0].weights.shape[1]:
			raise ValueError(f"Input size {input.shape[0]} does not coincide neuro layer size {self.layers[0].weights.shape[1]}")
		output = input.copy()
		output = np.reshape(output, (-1, 1))
		for layer in self.layers:
			output = layer.CalculateOutput(output)
			output = Sigmoid(output)
		return output
	def Backward(self, input, target):
		# print("Back... 1/2", end='')
		# Backpropagation (output layer)
		deltaOuput =  ((input - target))
		i = len(self.layers)-1
		for layer in reversed(self.layers):
			tmp = (layer.weights.transpose() @ SigmoidDerivative(layer.neuroValues))
			layer.weights -= self.learningRate * tmp.transpose() * deltaOuput * 2
			layer.bias    -= self.learningRate * deltaOuput * SigmoidDerivative(layer.neuroValues) * 2
			if i == 0:
				tmp = SigmoidDerivative(self.input)	
			else:
				tmp = SigmoidDerivative(self.layers[i-1].neuroValues)
			deltaOuput = layer.weights.transpose() @ deltaOuput * tmp
			i -= 1
			# delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
	def Train(self, learningRate = .1, epochs = 3):
		self.learningRate = learningRate
		images, labels = utils.load_dataset()

		e_loss = 0
		e_correct = 0
		for epoch in range(epochs):
			print(f"Epoch #{epoch}")
			numImages = labels.shape[0]
			for i, (image, label) in enumerate(zip(images, labels)):
				sys.stdout.write(f"{round(float(i)/numImages*100)}%\r")
				# image = np.reshape(image, (-1, 1))
				label = np.reshape(label, (-1, 1))

				output = self.Forward(image)
				self.Backward(output, label)
				e_loss += ((np.sum(np.sqrt((output - label) ** 2), axis = 1)) / numImages)[0]
				e_correct += int(np.argmax(output) == np.argmax(label))
				sys.stdout.write(f"{round(float(i)/numImages*100)}%. Loss: {round((e_loss / (i+1)) * 100, 3)}%. Accuracy: {round((e_correct / (i+1)) * 100)}%    \r")
			# print some debug info between epochs
			print(f"Loss: {round((e_loss / numImages) * 100, 3)}%                         ")
			print(f"Accuracy: {round((e_correct / numImages) * 100, 3)}%")
			e_loss = 0
			e_correct = 0
	def Guess(self, test_image):
		# CHECK CUSTOM

		# Grayscale + Unit RGB + inverse colors
		gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
		test_image = 1 - (gray(test_image).astype("float32") / 255)

		# Reshape
		test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

		# Predict
		image = np.reshape(test_image, (-1, 1))
		guess = self.Forward(image).argmax()

		# plt.imshow(test_image.reshape(28, 28), cmap="Greys")
		# plt.title(f"NN suggests that the number is: {guess}")
		# plt.show()
		return guess

test_image = plt.imread("custom.jpg", format="jpeg")
images, labels = utils.load_dataset()
inpuNeuronsNum = images.shape[1]
outputNeuronsNum = labels.shape[1]
# Run()
myNetwork = NeuroWork(1,inpuNeuronsNum, outputNeuronsNum)
myNetwork.Train()
print(f"I think that 3 is {myNetwork.Guess(test_image)}")
# print("vot")

legacy.Run()