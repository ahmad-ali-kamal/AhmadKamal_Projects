def classifier(image_path, model_name):
    """
    Simulated classifier function for testing purposes.
    Replace with actual CNN classifier when available.
    """
    pet_name = image_path.split('/')[-1].lower()
    if "dog" in pet_name or "poodle" in pet_name or "beagle" in pet_name or "terrier" in pet_name:
        return "dog"
    else:
        return "not-a-dog"
