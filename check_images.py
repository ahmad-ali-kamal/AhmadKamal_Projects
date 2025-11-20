# Ahmed Ali Kamal
# Date: 2025-09-24

from time import time
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classifier import classifier
from print_results import print_results 

def main():
    in_arg = get_input_args()
    print(f"Image Folder: {in_arg.dir}, Model: {in_arg.arch}, Dog File: {in_arg.dogfile}")

    import os
    if not os.path.exists(in_arg.dir):
        print(f"Error: Folder {in_arg.dir} does not exist!")
        return

    start_time = time()

    results_dic = get_pet_labels(in_arg.dir)

    models = ['vgg', 'resnet', 'alexnet']
    all_results = {}

    for model in models:
        model_results = {}
        for key in results_dic:
            pet_label = results_dic[key][0]
            classifier_label = classifier(in_arg.dir + key, model)
            match = 1 if pet_label == classifier_label else 0
            is_dog = 1 if pet_label in ["dog", "poodle", "beagle", "terrier"] else 0
            classifier_is_dog = 1 if classifier_label in ["dog"] else 0
            model_results[key] = [pet_label, classifier_label, match, is_dog, classifier_is_dog]
        all_results[model] = model_results

    end_time = time()
    tot_time = end_time - start_time
    print("\nTotal Runtime: {:02d}:{:02d}:{:02d}".format(int(tot_time/3600),
                                                        int((tot_time%3600)/60),
                                                        int((tot_time%3600)%60)))

    for model in models:
        print_results(model, all_results[model])

if __name__ == "__main__":
    main()
