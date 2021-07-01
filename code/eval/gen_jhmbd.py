

with open('jhmdb_imglist.txt') as f:
    imglist = f.readlines()

with open('jhmdb_lbllist.txt') as f:
    lbllist = f.readlines()

with open('jhmdb_split1.txt') as f:
    splitlist = f.readlines()

imglist = [img.strip() for img in imglist]
lbllist = [lbl.strip() for lbl in lbllist]

val_split = []
for vid_split in splitlist:
    vid = vid_split.strip()
    vid, is_val = vid.split(' ')
    if is_val == '2':
        val_split.append(vid.replace('.avi', ''))


for label in lbllist:
    components = label.split('/')
    iden = components[6]

    # Check iden in validation set
    found = False
    for val in val_split:
        if val == iden:
            found = True
            break

    if not found:
        continue
    
    for img in imglist:
        img_components = img.split('/')
        if iden == img_components[6]:
            print(img, label)
