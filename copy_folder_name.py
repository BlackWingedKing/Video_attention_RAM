import os 
# for (root,dirs,files) in os.walk('../'): 
#     print (root )
    # print (dirs )
    # print (files )
    # print ('--------------------------------')

# def walklevel(some_dir="../data/UCF-101", level=1):
#     print("num_sep")
#     some_dir = some_dir.rstrip(os.path.sep)
#     assert os.path.isdir(some_dir)
#     num_sep = some_dir.count(os.path.sep)
#     for root, dirs, files in os.walk(some_dir):
#         yield root, dirs, files
#         num_sep_this = root.count(os.path.sep)
#         if num_sep + level <= num_sep_this:
#             del dirs[:]

# print("b")
# a =walklevel(some_dir="../data/UCF-101", level=1)

a =os.listdir("../data/UCF-101/UCF-101")
for z in a:
	print(z)
	os.mkdir("../data/UCF-101/16_frames_split/complete_data/{}".format(str(z)))
	# print(os.cwd())

