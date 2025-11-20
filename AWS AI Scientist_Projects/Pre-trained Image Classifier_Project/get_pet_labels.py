from os import listdir

def get_pet_labels(image_dir):
    filename_list = listdir(image_dir)
    results_dic = dict()
    for filename in filename_list:
        if filename.startswith("."):  
            continue
        lower_filename = filename.lower()
        word_list = lower_filename.split("_")
        pet_label = ""
        for word in word_list:
            if word.isalpha():
                pet_label += word + " "
        pet_label = pet_label.strip()
        if filename not in results_dic:
            results_dic[filename] = [pet_label]
    return results_dic
