def print_results(model_name, results_dic):
    total_images = len(results_dic)
    n_dogs_img = sum([1 for key in results_dic if results_dic[key][3] == 1])
    n_notdogs_img = total_images - n_dogs_img
    correct_dogs = sum([1 for key in results_dic if results_dic[key][3] == 1 and results_dic[key][4] == 1])
    correct_notdogs = sum([1 for key in results_dic if results_dic[key][3] == 0 and results_dic[key][4] == 0])
    correct_breed = sum([1 for key in results_dic if results_dic[key][2] == 1 and results_dic[key][3] == 1])

    print("\n" + "="*60)
    print(f"Results for CNN Model: {model_name.upper()}")
    print("="*60)
    print(f"Total Images: {total_images}")
    print(f"Number of Dog Images: {n_dogs_img}")
    print(f"Number of 'Not-a-Dog' Images: {n_notdogs_img}")
    print(f"Correct Dog Classification: {correct_dogs} ({correct_dogs/n_dogs_img*100 if n_dogs_img>0 else 0:.1f}%)")
    print(f"Correct 'Not-a-Dog' Classification: {correct_notdogs} ({correct_notdogs/n_notdogs_img*100 if n_notdogs_img>0 else 0:.1f}%)")
    print(f"Correct Dog Breed Classification: {correct_breed} ({correct_breed/n_dogs_img*100 if n_dogs_img>0 else 0:.1f}%)")

    print("\nFirst 10 Classified Images:")
    for idx, key in enumerate(list(results_dic.keys())[:10]):
        pet_label, classifier_label, match, is_dog, classifier_is_dog = results_dic[key]
        print(f"{idx+1:2d} File: {key:25} Pet Label: {pet_label:20} Classifier: {classifier_label:20} Match: {match}")
