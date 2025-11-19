import os

image_dir = '/home/sheli/Projects/DBow3/utils/images'
image_list = os.listdir(image_dir)
for i in range(len(image_list)):
    old_name = os.path.join(image_dir, image_list[i])
    new_name = os.path.join(image_dir, 'image{}.png'.format(i))
    os.renames(old_name, new_name)

print('Done')