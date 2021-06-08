

with open('jhmdb_imglist.txt') as f:
    imglist = f.readlines()

with open('jhmdb_lbllist.txt') as f:
    lbllist = f.readlines()

imglist = [img.strip() for img in imglist]
lbllist = [lbl.strip() for lbl in lbllist]

for label in lbllist:
    components = label.split('/')
    iden = components[5]
    
    for img in imglist:
        img_components = img.split('/')
        if iden == img_components[5]:
            print(img, label)

