from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
from numbers import Number

class DenseCRF:
    def __init__(self, nvar, nlabels, width, height, ann):
        self.nvar = nvar
        self.ann = ann
        self.nlabels = nlabels
        self.width = width
        self.height = height
        self.pair_wize_function = None
        self.unary_function = None

    def setUnaryEnergy(self, unary):
        print("unary setted")
        self.unary_function = unary

    def addPairwiseEnergy(self, pair, compat):
        print("added")
        if self.pair_wize_function is None:
            self.pair_wize_function = pair
        else:
            self.pair_wize_function = np.concatenate((self.pair_wize_function, pair), 0)

    def exp_normalizer(self, inp):
        print("initializing")
        temp = np.subtract(inp, np.max(inp, 0))
        print(temp)
        print(temp.shape)
        tempo = np.exp(temp)
        normalized = np.divide(tempo, np.sum(tempo))
        return normalized

    def inference(self, n_step):
        print("start infering:")
        Q = self.exp_normalizer(-self.unary_function)
        print(Q.shape)
        print(Q)
        print(self.pair_wize_function.shape)
        mu = np.ones((self.nvar,))
        for i in range(self.height):
            for j in range(self.width):
                if self.ann[i, j] is 1:#moteghayera vali injori nist :|
                    mu[i*self.height+j] = 0
        print('mu')
        print(mu)
        print(mu.shape)
        for i in range(n_step):
            temp1 = -self.unary_function
            print('temp1')
            print(temp1.shape)
            for k in range(self.pair_wize_function.shape[0]):
                temp2 = self.apply(Q, self.pair_wize_function[k])
                temp3 = np.multiply(temp2, mu)
                temp1 = np.subtract(temp1, temp3)
            Q = self.exp_normalizer(temp1)
        return Q

    def apply(self, Q, pair_row):
        print("apply")
        print(Q.shape)
        print(pair_row.shape)
        print(np.dot(Q, pair_row))
        Q_bar = np.multiply(Q, pair_row)
        print(Q_bar)
        print(Q_bar.shape)
        print("end of applying")
        return Q_bar


root = 'C:/Users/Hamed Khashehchi/Downloads/Compressed/show me bobs/densecrf/densecrf/examples/'
drat = 'C:/Users/Hamed Khashehchi/Desktop/kha/'
# root = drat
number_classes = 21
ground_truth_probability = 0.5


def get_color(pixel):
    return pixel[0]+256*pixel[1]+256*256*pixel[2]


def zero_list_maker(n):
    list_of_zeros = [0] * n
    return list_of_zeros


def unary_from_labels(labels, n_labels, gt_prob, zero_unsure=True):

    assert 0 < gt_prob < 1, "`gt_prob must be in (0,1)."

    labels = labels.flatten()

    n_energy = -np.log((1.0 - gt_prob) / (n_labels - 1))
    p_energy = -np.log(gt_prob)

    unary_function = np.full((n_labels, len(labels)), n_energy, dtype='float32')
    unary_function[labels - 1 if zero_unsure else labels, np.arange(unary_function.shape[1])] = p_energy

    if zero_unsure:
        unary_function[:, labels == 0] = -np.log(1.0 / n_labels)

    return unary_function


def create_pairwise_gaussian(sdims, shape):
    # create mesh
    print("gaussian")
    print(shape)
    hcord_range = [range(s) for s in shape]
    print(hcord_range)
    mesh = np.array(np.meshgrid(*hcord_range, indexing='ij'), dtype=np.float32)

    # scale mesh accordingly
    for i, s in enumerate(sdims):
        print(i)
        print(s)
        mesh[i] /= s
    return mesh.reshape([len(sdims), -1])


def create_pairwise_bilateral(sdims, schan, img, chdim=-1):
    print("bilateral")
    if chdim == -1:
        im_feat = img[np.newaxis].astype(np.float32)
    else:
        im_feat = np.rollaxis(img, chdim).astype(np.float32)

    if isinstance(schan, Number):
        im_feat /= schan
    else:
        for i, s in enumerate(schan):
            im_feat[i] /= s

    cord_range = [range(s) for s in im_feat.shape[1:]]
    mesh = np.array(np.meshgrid(*cord_range, indexing='ij'), dtype=np.float32)

    for i, s in enumerate(sdims):
        mesh[i] /= s

    feats = np.concatenate([mesh, im_feat])
    return feats.reshape([feats.shape[0], -1])


image = imread(root+'/im1.ppm')
print(image.shape)
height_image, width_image, channel_image = image.shape
annotation = imread(root+'/anno1.ppm')
print(annotation.shape)
print(annotation.dtype)
height_annotation, width_annotation, channel_annotation = annotation.shape
annotation_label = annotation[:, :, 0] + (256*annotation[:, :, 1]) + (256*256*annotation[:, :, 1])
print(annotation_label)
colors, labels = np.unique(annotation_label, return_inverse=True)
is_there_anything_unknown = 0 in colors
print(colors)
print(labels)
print(is_there_anything_unknown)
if is_there_anything_unknown:
    colors = colors[1:]
color_table = np.empty((len(colors), 3))
color_table[:, 0] = (colors & 0x0000FF)
color_table[:, 1] = (colors & 0x00FF00) / 256
color_table[:, 2] = (colors & 0xFF0000) / (256*256)
n_labels = len(set(labels.flat)) - int(is_there_anything_unknown)
print(n_labels, " labels", (" plus \"unknown\" 0: " if is_there_anything_unknown else ""), set(labels.flat))

fig, axs = plt.subplots(1, 2)
axs[0].set_title('Image:')
axs[1].set_title('Annotation:')
axs[0].imshow(image)
axs[1].imshow(annotation)
plt.show()

print("done showing off")

d = DenseCRF(width_image*height_image, n_labels, width_image, height_image, annotation_label)

unary_functions = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=is_there_anything_unknown)
d.setUnaryEnergy(unary=unary_functions)

print(unary_functions)
print(unary_functions.shape)
feats = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
print(feats)
print(feats.shape)
d.addPairwiseEnergy(pair=feats, compat=3)

print("popopopop")
print(d.pair_wize_function.shape)


feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13), img=image, chdim=2)
print(feats)
print(feats.shape)
d.addPairwiseEnergy(pair=feats, compat=10)

print("popopopop")
print(d.pair_wize_function.shape)

Q = d.inference(n_step=10)
print("end of inference")
print(Q)
print(Q.shape)

MAP = np.argmax(Q, axis=0)
MAP = color_table[MAP, :]
MAP = MAP.reshape((height_image, width_image, channel_image))
print(MAP)
print(MAP.shape)
fig, axs = plt.subplots(1, 2)
axs[0].set_title('Image:')
axs[1].set_title('OUR Segmentation:')
axs[0].imshow(image)
axs[1].imshow(MAP, 'Blues_r')
plt.show()
