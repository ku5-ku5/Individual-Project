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
	train(masked_model, lossFn, mask_optimiser, masked_classes, mask_train, 1)#, "masked_model")

	#train unmasked model
	train(unmasked_model, lossFn, face_optimiser, maskless_classes, face_train, 1)#, "unmasked_model")


	print('Training Complete')

	#test masked model
	masked_matrix1 = confusion_matrix(masked_model, mask_test)

	#test unmasked model
	unmasked_matrix1 = confusion_matrix(unmasked_model, face_test)

	#test both models with opposite datasets
	masked_matrix2 = confusion_matrix(masked_model, face_test)
	unmasked_matrix2 = confusion_matrix(unmasked_model, mask_test)

	matrices = {"masked_results" : [masked_matrix1, masked_matrix2], "unmasked_results" : [unmasked_matrix1 ,unmasked_matrix2]}

	def calc_stats(conf_matrix):
		#calculate precision
		p = Precision(conf_matrix[0][0], conf_matrix[1][0])


		#calculate recall
		r = Recall(conf_matrix[0][0], conf_matrix[0][1])

		#calculate f1 score
		f1 = F1(conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0])

		return(p, r, f1)

	for key in matrices.keys():
		file = open(key + ".txt", 'a')
		out = ""
		for i in matrices[key]:
			p, r, f1 = calc_stats(i)
			conf_matrix = i
			out += "\nTesting Results:\n"
			out += "\nTrue Positives: " + str(conf_matrix[0][0]) + "\nFalse Positives: " + str(conf_matrix[0][1]) + "\nFalse Negatives: " + str(conf_matrix[1][0]) + "\nTrue Negatives: " + str(conf_matrix[1][1]) + "\nPrecision: " + str(p) + "\nRecall: " + str(r) + "\nF1: " + str(f1)
			#print("\nTrue Positives: " + str(conf_matrix[0][0]) + "\nFalse Positives: " + str(conf_matrix[0][1]) + "\nFalse Negatives: " + str(conf_matrix[1][0]) + "\nTrue Negatives: " + str(conf_matrix[1][1]))
			#print("Precision: " + str(p))
			#print("Recall: " + str(r))
			#print("F1: " + str(f1))
		print(out)
		file.write(out)
		file.close()






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