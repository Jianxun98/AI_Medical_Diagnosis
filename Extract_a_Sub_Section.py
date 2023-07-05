import numpy as np
import keras
import pandas as pd

# Define a simple one dimensional "image" to extract from
image = np.array([10,11,12,13,14,15])
image

# Compute the dimensions of your "image"
image_length = image.shape[0]
image_length

# Define a patch length, which will be the size of your extracted sub-section
patch_length = 3

# Define your start index
start_i = 0

# Define an end index given your start index and patch size
print(f"start index {start_i}")
end_i = start_i + patch_length
print(f"end index {end_i}")

# Extract a sub-section from your "image"
sub_section = image[start_i: end_i]
print("output patch length: ", len(sub_section))
print("output patch array: ", sub_section)

# Add one to your start index
start_i +=1

# Set your start index to 3 to extract a valid patch
start_i = 3
print(f"start index {start_i}")
end_i = start_i + patch_length
print(f"end index {end_i}")
sub_section = image[start_i: end_i]
print("output patch array: ", sub_section)

# Compute and print the largest valid value for start index
print(f"The largest start index for which "
      f"a sub section is still valid is "
      f"{image_length - patch_length}")

# Compute and print the range of valid start indices
print(f"The range of valid start indices is:")

# Compute valid start indices, note the range() function excludes the upper bound
valid_start_i = [i for i in range(image_length - patch_length + 1)]
print(valid_start_i)

# Choose a random start index, note the np.random.randint() function excludes the upper bound.
start_i = np.random.randint(image_length - patch_length + 1)
print(f"randomly selected start index {start_i}")

# Randomly select multiple start indices in a loop
for _ in range(10):
    start_i = np.random.randint(image_length - patch_length + 1)
    print(f"randomly selected start index {start_i}")

# We first simulate input data by defining a random patch of length 16. This will contain labels 
# with the categories (0 to 3) as defined above.

patch_labels = np.random.randint(0, 4, (16))
print(patch_labels)

# A straightforward approach to get the background ratio is
# to count the number of 0's and divide by the patch length

bgrd_ratio = np.count_nonzero(patch_labels == 0) / len(patch_labels)
print("using np.count_nonzero(): ", bgrd_ratio)
print(np.count_nonzero(patch_labels == 0))
print(len(patch_labels))
bgrd_ratio = len(np.where(patch_labels == 0)[0]) / len(patch_labels)
print("using np.where(): ", bgrd_ratio)
print( len(np.where(patch_labels == 0)[0]))

# However, take note that we'll use our label array to train a neural network
# so we can opt to compute the ratio a bit later after we do some preprocessing. 
# First, we convert the label's categories into one-hot format so it can be used to train the model

patch_labels_one_hot = keras.utils.to_categorical(patch_labels, num_classes=4)
print(patch_labels_one_hot)
print(len(patch_labels_one_hot))

# Let's convert the output to a dataframe just so we can see the labels more clearly

pd.DataFrame(patch_labels_one_hot, columns=['background', 'edema', 'non-enhancing tumor', 'enhancing tumor'])

# What we're interested in is the first column because that 
# indicates if the element is part of the background
# In this case, 1 = background, 0 = non-background

print("background column: ", patch_labels_one_hot[:,0])

# we can compute the background ratio by counting the number of 1's 
# in the said column divided by the length of the patch

bgrd_ratio = np.sum(patch_labels_one_hot[:,0])/ len(patch_labels)
print("using one-hot column: ", bgrd_ratio)
