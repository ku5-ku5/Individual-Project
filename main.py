from training import *
from testing import *


if __name__ == '__main__':
	#Load the image data in

	#define the possible classes for each model
	masked_classes = ['negative_data', 'with_mask']
	maskless_classes = ['negative_data', 'without_mask']

	#create test and train dataloaders for each model
	mask_train, mask_test = retrieveData("./data/training/mask_training", masked_classes, 6851)
	face_train, face_test = retrieveData("./data/training//maskless_training", maskless_classes, 6954)

	#epochs = input("Enter number of epochs: ")

	#defining neural networks
	masked_model = Net()
	masked_model.to(device)

	unmasked_model = Net()
	unmasked_model.to(device)

	#defining optimisers
	mask_optimiser = optim.SGD(masked_model.parameters(), lr=0.001, momentum=0.9)
	face_optimiser = optim.SGD(unmasked_model.parameters(), lr=0.001, momentum=0.9)

	#defining a loss function
	lossFn = nn.CrossEntropyLoss()

	#train masked model
	train(masked_model, lossFn, mask_optimiser, masked_classes, mask_train, 5, "masked_model")

	#train unmasked model
	train(unmasked_model, lossFn, face_optimiser, maskless_classes, face_train, 5, "unmasked_model")


	print('Training Complete')

	#test masked model


	#test unmasked model

'''
	conf_matrix = confusion_matrix(net, testData)

	p = Precision(conf_matrix[0][0], conf_matrix[1][0])

	r = Recall(conf_matrix[0][0], conf_matrix[0][1])

	f1 = F1(conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0])

	print("\nTrue Positives: " + str(conf_matrix[0][0]) + "\nFalse Positives: " + str(conf_matrix[0][1]) + "\nFalse Negatives: " + str(conf_matrix[1][0]) + "\nTrue Negatives: " + str(conf_matrix[1][1]))
	print("Precision: " + str(p))
	print("Recall: " + str(r))
	print("F1: " + str(f1))
'''