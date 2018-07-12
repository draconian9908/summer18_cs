# ## Homework _9_ (not 8): Problem 2, steganography
#
# This question asks you to write two functions, likely with some helper functions, that will enable you
# to embed arbitrary text (string) messages into an image (if there is enough room!)

# For extra credit, the challenge is to be
# able to extract/embed an image into another image...

#
# You'll want to borrow from hw8pr1 for
#  + opening the file
#  + reading the pixels
#  + create some helper functions!
#  + also, check out the slides :-)
#
# Happy steganographizing, everyone!
#
import matplotlib.pyplot as plt
import cv2

def get_bits(image):
    bits = []
    num_rows, num_cols, num_channels = image.shape
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = image[row,col]
            r_bin = bin(r); g_bin = bin(g); b_bin = bin(b)
            bits.append(r_bin[-1])
            bits.append(g_bin[-1])
            bits.append(b_bin[-1])
    return bits
# Part A: here is a signature for the decoding
# remember - you will want helper functions!
def desteg_string( image ):
    bits = get_bits(image)
    full = ''.join(bits)
    tot_len = len(full)
    chars = []
    for i in range(0,tot_len,8):
        piece = full[i:i+8]
        if piece == '00000000':
            break
        num = int(piece, 2)
        char = chr(num)
        chars.append(char)
    message = ''.join(chars)
    print(message)





def string_to_binary(message):
    bits = []
    for char in message:
        el = ord(char)
        bi = bin(el)
        bi = bi.lstrip('0b')
        needed = len(bi)
        addi = 8-needed
        bi = ('0' * addi) + bi
        bits.append(bi)
    bix = ''.join(bits)
    return bix

def apply_to_pix(new_image, bix):
    num_rows, num_cols, num_channels = new_image.shape
    count = 0
    for row in range(num_rows):
        for col in range(num_cols):
            r, g, b = new_image[row,col]
            if count < len(bix):
                rb = bix[count]
                count += 1
                nr = convert_for_pix(r,rb)
                if count < len(bix):
                    gb = bix[count]
                    count += 1
                    ng = convert_for_pix(g,gb)
                    if count < len(bix):
                        bb = bix[count]
                        count += 1
                        nb = convert_for_pix(b,bb)
                    else:
                        bb = '0'
                        count += 1
                        nb = convert_for_pix(b,bb)
                else:
                    gb = '0'
                    count += 1
                    ng = convert_for_pix(g,gb)
                    bb = '0'
                    count += 1
                    nb = convert_for_pix(b,bb)
                new_image[row,col] = [nr,ng,nb]
            elif count < len(bix)+16:
                rb = '0'
                count += 1
                nr = convert_for_pix(r,rb)
                if count < len(bix)+16:
                    gb = '0'
                    count += 1
                    ng = convert_for_pix(g,gb)
                    if count < len(bix)+16:
                        bb = '0'
                        count += 1
                        nb = convert_for_pix(b,bb)
                    else:
                        nb = b
                else:
                    ng = g
                    nb = b
                new_image[row,col] = [nr,ng,nb]
            else:
                break
    return new_image


def convert_for_pix(pix, fix):
    pix_bin = bin(pix)
    pix_bl = list(pix_bin)
    pix_bl[-1] = fix; pix_bl[1] = '0'
    pix_bs = "".join(pix_bl)
    n_pix = int(pix_bs, 2)
    return n_pix

def to_file(edited_image):
    global outname
    new_image = cv2.cvtColor(edited_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outname, new_image)
# Part B: here is a signature for the encoding/embedding
# remember - you will want helper functions!
def steganographize( image, message ):
    """ be sure to include a better docstring here! """
    new_image = image.copy()
    bix = string_to_binary(message)
    edited_image = apply_to_pix(new_image, bix)
    to_file(edited_image)


def main(filename, outname):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # desteg_string(image)
    # message = "Follow. But! Follow only if you be men of valor, for the entrance to this cave is guarded by a creature so foul, so cruel that no man yet has fought with it and lived!Bones of full 50 men lie strewn about its lair, so brave knights, if you do doubt your courage or your strength, come no further! For death awaits you all with nasty, big, pointy teeth."
    message = "Follow. But! Follow only if you be men of valor, for the entrance to this cave is guarded by a creature so foul, so cruel that no man yet has fought with it and lived! Bones of full 50 men lie strewn about its lair, so brave knights, if you do doubt your courage or your strength, come no further! For death awaits you all with nasty, big, pointy teeth."
    # message = "test"
    print("Message sent is:\n" + message)
    steganographize(image, message)
    image = cv2.imread(outname, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("\nMessage translated is:")
    desteg_string(image)


if __name__ == "__main__":
    filename = "cursed2.JPG"
    outname = filename[:-4] + "_out.png"
    main(filename, outname)
