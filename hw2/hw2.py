import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

f = open("logfile.txt", "w")


class image_file:
    def __init__(self, file_pos, gray_flag):
        if gray_flag:
            self.image = cv2.imread(file_pos, cv2.IMREAD_GRAYSCALE)
        else:
            self.image = cv2.imread(file_pos)

        f.writelines('image shape: ' + str(self.image.shape) + '\nimage type: ' + str(type(self.image)) + '\n')
    def make_mask(self):
        M, N = self.image.shape[0] ,  self.image.shape[1]
        self.mask = (self.image == 255)
        f.writelines('mask shape: ' + str(self.mask.shape) + '\nmask type: ' + str(type(self.mask)) + '\n')
        f.writelines('MASK    true: ' + str(np.count_nonzero(self.mask == True) / (M*N)) + '    False: ' + str(np.count_nonzero(self.mask == False) / (M*N)) + '\n')
        #f.writelines('image mask: ' + str(self.mask))
    def reshape_image(self):
        M, N, O = self.image.shape[0] ,  self.image.shape[1], self.image.shape[2]
        self.image = np.reshape(self.image, (M*N, O))
        self.M = M
        self.N = N
        self.O = O
        f.writelines('image shape: ' + str(self.image.shape) + '\nimage type: ' + str(type(self.image)) + '\n')


def create_gmm(components_num, input_image):
    gmm = GaussianMixture(n_components=components_num, random_state=0).fit(input_image.image)
    f.writelines('gmm.means_ shape: ' + str(gmm.means_.shape) + '\ngmm.means_ type: ' + str(type(gmm.means_)) + '\n')
    f.writelines('gmm.means: ' + str(gmm.means_) + '\n')

    return gmm

def predict_gmm(gmm, input_image):
    M, N = input_image.M ,  input_image.N
    f.writelines("M: " + str(M) + '  N: ' + str(N) + '\n')
    predict_answer = gmm.predict(input_image.image)
    predict_answer = predict_answer.reshape(M, N)

    f.writelines('gmm.predict shape: ' + str(predict_answer.shape) + '\ngmm.predict type: ' + str(type(predict_answer)) + '\n')
    return predict_answer

def show_predict_image(gmm, predict_answer, image_name):
    #f.writelines('gmm.means_[predict_answer]: ' + str(gmm.means_[predict_answer]) + '\n')
    #f.writelines('gmm.means_[predict_answer].shape: ' + str(gmm.means_[predict_answer].shape) + '\n')

    cv2.imshow(image_name, gmm.means_[predict_answer].astype('int8'))
    cv2.waitKey(0)

def get_accuracy(predict_answer, input_mask, components_num):
    #f.writelines('predict_answer == 0: ' + str( np.count_nonzero(predict_answer == 1)) + '\n')
    #f.writelines(str((predict_answer)))
    pixel_num_array = np.array([np.count_nonzero(predict_answer == i) for i in range(0, components_num)])

    arg = np.argsort(pixel_num_array)
    print('pixel_num_array: ' + str(pixel_num_array) + '  arg: '+ str(arg) + '\n')

    for i in range(0, components_num):
        if i < ((components_num-1)/2):
            predict_answer = np.where(predict_answer == arg[i], -1, predict_answer)
        else:
            predict_answer = np.where(predict_answer == arg[i], -2, predict_answer)

    predict_answer = np.where(predict_answer == -1, False, predict_answer)
    predict_answer = np.where(predict_answer == -2, True, predict_answer)

    #exit(1)
    
    corret_num = np.count_nonzero((predict_answer == input_mask.mask) == True)
    all_num = predict_answer.shape[0] * predict_answer.shape[1]
    precision = corret_num/all_num

    f.writelines('corret_num: ' + str(corret_num) + '  all_num: ' + str(all_num) + '  precision: ' + str(precision) + '\n')
    #f.writelines('predict_aswer: ' + str((predict_answer)) + '\n')

    return predict_answer, precision

def draw_figure(q_num, xlim_array, ylim_array, ylim_label, ylim_array_2, ylim_label_2, flag_2):
    plt.figure(1)
    plt.title('Question ' + str(q_num) +  '  precision')
    plt.xlabel('Mixture number')
    plt.ylabel('Precision')
    #plt.xlim(xlim_array)
    #plt.ylim(ylim_array)

    plt.plot(xlim_array, ylim_array, label=ylim_label)
    if flag_2 == True:
        plt.plot(xlim_array, ylim_array_2, label=ylim_label_2)

    plt.legend()
    plt.savefig('Question ' + str(q_num) +  '  precision' + ".png")

def run_question(q_num):
    range_min = 2
    range_max = 25

    if(q_num == 1):
        
        input_mask = image_file('soccer1_mask.png', True)
        #input_mask.reshape_image()
        input_mask.make_mask()

        input_image = image_file('soccer1.jpg', False)
        input_image.reshape_image()

        precision_array = np.array([])
        for components_num in range(range_min, range_max):
            f.writelines('-----------components_num:  ' + str(components_num) + '------------------------\n')
            gmm  = create_gmm(components_num, input_image)
            predict_answer = predict_gmm(gmm, input_image)

            predict_answer, precision = get_accuracy(predict_answer, input_mask, components_num)

            precision_array = np.append(precision_array, precision)

            #show_predict_image(gmm, predict_answer, 'image')
            f.writelines('----------------------------------------------------------------------\n')

        f.writelines('precision_array: ' + str((precision_array)) + '\n')
        draw_figure(q_num, np.array([k for k in range(range_min, range_max)]), precision_array, 'accuracy of soccer1', None, None, False)
        
    elif(q_num == 2):
        input_mask = image_file('soccer2_mask.png', True)
        #input_mask.reshape_image()
        input_mask.make_mask()

        build_image = image_file('soccer1.jpg', False)
        build_image.reshape_image()

        test_image = image_file('soccer2.jpg', False)
        test_image.reshape_image()

        precision_array = np.array([])
        for components_num in range(range_min, range_max):
            f.writelines('-----------components_num:  ' + str(components_num) + '------------------------\n')
            gmm  = create_gmm(components_num, build_image)
            predict_answer = predict_gmm(gmm, test_image)

            predict_answer, precision = get_accuracy(predict_answer, input_mask, components_num)

            precision_array = np.append(precision_array, precision)

            #show_predict_image(gmm, predict_answer, 'image')
            f.writelines('----------------------------------------------------------------------\n')

        f.writelines('precision_array: ' + str((precision_array)) + '\n')
        draw_figure(q_num, np.array([k for k in range(range_min, range_max)]), precision_array, 'accuracy of soccer2', None, None, False)

    elif(q_num == 3):
        input_mask_1 = image_file('soccer1_mask.png', True)
        input_mask_1.make_mask()

        input_mask_2 = image_file('soccer2_mask.png', True)
        input_mask_2.make_mask()

        input_image_1 = image_file('soccer1.jpg', False)
        input_image_1.reshape_image()

        input_image_2 = image_file('soccer2.jpg', False)
        input_image_2.reshape_image()

        build_image = image_file('soccer1.jpg', False)
        build_image.image = np.concatenate((input_image_1.image, input_image_2.image), axis=0)

        #print(build_image.image.shape)
        #exit(1)

        precision_array_1 = np.array([])
        precision_array_2 = np.array([])
        for components_num in range(range_min, range_max):
            f.writelines('-----------components_num:  ' + str(components_num) + '------------------------\n')
            gmm  = create_gmm(components_num, build_image)

            predict_answer_1 = predict_gmm(gmm, input_image_1)
            predict_answer_1, precision_1 = get_accuracy(predict_answer_1, input_mask_1, components_num)
            precision_array_1 = np.append(precision_array_1, precision_1)

            predict_answer_2 = predict_gmm(gmm, input_image_2)
            predict_answer_2, precision_2 = get_accuracy(predict_answer_2, input_mask_2, components_num)
            precision_array_2 = np.append(precision_array_2, precision_2)



            #show_predict_image(gmm, predict_answer, 'image')
            f.writelines('----------------------------------------------------------------------\n')

        f.writelines('precision_array_1: ' + str((precision_array_1)) + '\n')
        f.writelines('precision_array_2: ' + str((precision_array_2)) + '\n')
        draw_figure(q_num, np.array([k for k in range(range_min, range_max)]), precision_array_1, 'accuracy of soccer1', precision_array_2, 'accuracy of soccer2', True)





if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    run_question(3)
    

