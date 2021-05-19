from training import *
from testing import *


if __name__ == '__main__':
	#Load the image data in

	#define the possible classes for each model
	masked_classes = ['negative_data', 'with_mask']
	maskless_classes = ['negative_data', 'without_mask']

	#create test and train dataloaders for each model
	mask_train, mask_test = retrieveData("./data/training/mask_training", masked_classes)
	
	face_train, face_test = retrieveData("./data/training//maskless_training", maskless_classes)
'''
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
	train(masked_model, lossFn, mask_optimiser, masked_classes, mask_train, 1, "Masked")

	#train unmasked model
	train(unmasked_model, lossFn, face_optimiser, maskless_classes, face_train, 1, "Unmasked")


	print('Training Complete')

	#test masked model with masked dataset
	masked_mask = confusion_matrix(masked_model, mask_test)

	#test unmasked model with masked dataset
	unmasked_mask = confusion_matrix(unmasked_model, mask_test)

	#test both models with unmasked dataset
	masked_face = confusion_matrix(masked_model, face_test)
	unmasked_face = confusion_matrix(unmasked_model, face_test)

	matrices = {"Masked" : [masked_mask, masked_face], "Unmasked" : [unmasked_mask ,unmasked_face]}

	def calc_stats(conf_matrix):
		#calculate precision
		p = Precision(conf_matrix[0][0], conf_matrix[1][1])

		#calculate recall
		r = Recall(conf_matrix[0][0], conf_matrix[0][1])

		#calculate f1 score
		f1 = F1(conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][1])

		return(p, r, f1)

	for key in matrices.keys():
		file = open(key + "_results.txt", 'a')
		out = '#' * 25 + " " + key + " Testing Results " + '#' * 25 + "\n\n"
		for i in range(len(matrices[key])):
			conf_matrix = matrices[key][i]
			p, r, f1 = calc_stats(conf_matrix)
			if i == 0:
				out += "Masked Dataset\n\n"
			else:
				out += "Unmasked Dataset\n\n"
			out += "\nTrue Positives: " + str(conf_matrix[0][0]) + "\nFalse Positives: " + str(conf_matrix[0][1]) + "\nFalse Negatives: " + str(conf_matrix[1][1]) + "\nTrue Negatives: " + str(conf_matrix[1][0]) 
			out += "\nPrecision: " + str(p) + "\nRecall: " + str(r) + "\nF1: " + str(f1) + "\n"
		print(out)
		file.write(out + "\n\n")
		file.close()'''