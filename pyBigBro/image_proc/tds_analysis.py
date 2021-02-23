"""
Created on 20.11.2019
@author: Igor Zagorodnov and Sergey Tomin
"""

import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
from scipy import ndimage
import pickle
import copy
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from skimage.transform import warp, AffineTransform

class TDSImage:
    def __init__(self, filename=None):
        self.filename = filename
        self.raw_image = None # one image or list of images
        self.bg_image = None
        self.images = None
        self.image = None
        self.noise_thresh = 0.08
        self.noise_proj_thresh = 0.1
        self.charge = 250  # charge in [pC]
        self.dx_px = 0.0438912  # ps / px, pixel size in horizontal direction
        self.dy_px = 0.003260   # MeV / px, pixel size in vertical direction
        self.unif_filter_size = 100
        self.hw = 350
        self.gauss_filter_sigma = 0.5
        self.shear = 0. # Shear angle in counter-clockwise direction as radians.
        self.fliplr = False

    def load_image(self, filename):
        with open(filename, 'rb') as f:
            raw_image = pickle.load(f)
        return raw_image

    def read_image_from_pickle(self):
        # read image
        if self.filename is not None:
            self.raw_image = self.load_image(self.filename)
            return True
        else:
            print("filename is None")
            self.raw_image = None
            return False

    def load_images(self, filenames):
        images = []
        for filename in filenames:
            images.append(self.load_image(filename))
        return images


    def get_processed_image(self):
        if self.image is not None:
            return np.flipud(self.image)
        else:
            return None

    def shearing(self, image):
        X, Y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        xy = np.vstack((X.flatten(), Y.flatten()))

        transform = np.array([[1, np.tan(self.shear)],
                              [0, 1]])
        xy_new = np.dot(transform, xy)

        img_new = ndimage.map_coordinates(image, coordinates=[xy_new[1, :], xy_new[0, :]], output=None, order=1).reshape(
            image.shape)

        # tform = AffineTransform(shear=self.shear)
        # image = warp(img, tform.inverse, output_shape=img.shape)
        return img_new

    def center_images(self, images):
        if len(images) > 1:
            nimg = len(images)
            # find center of mass for cetering
            jj = np.array([])
            ii = np.array([])
            nxx = np.array([])
            nyy = np.array([])
            for img in images:
                (j, i) = ndimage.center_of_mass(img)
                # print(i, j)
                # plt.imshow(img, aspect='auto')
                # plt.show()
                ny, nx = np.shape(img)
                jj = np.append(jj, int(j))
                ii = np.append(ii, int(i))
                nyy = np.append(nyy, ny)
                nxx = np.append(nxx, nx)
            j0 = jj - np.min(jj)
            j1 = jj + np.min(nyy - (jj))
            i0 = ii - np.min(ii)
            i1 = ii + np.min(nxx - (ii))

            # images averaging
            center_imgs = []
            for i in range(nimg):
                img = images[i][int(j0[i]):int(j1[i]), int(i0[i]):int(i1[i])]

                center_imgs.append(img)
            return center_imgs

        else:
            return images

    def averaging_images(self, images):
        if len(images) > 1:
            av_img = np.mean(images, axis=0)
        else:
            av_img = images[0]
        return av_img

    def process_image(self, image):
        if self.bg_image is not None:
            img0 = image.astype(np.double) - self.bg_image.astype(np.double)
        else:
            img0 = image.astype(np.double)
        # in case after background removing was negative values

        inds_mask_neg = np.asarray(img0 < 0).nonzero()
        img0[inds_mask_neg] = 0.0

        img0 = np.flipud(img0)
        img1 = ndimage.uniform_filter(img0, size=self.unif_filter_size)
        (ny, nx) = img0.shape

        # clip image
        (j0, i0) = np.unravel_index(np.argmax(img1, axis=None), img1.shape)

        j1 = max(j0 - self.hw * 5, 0)
        j2 = min(j0 + self.hw * 5, ny)
        i1 = max(i0 - self.hw * 3, 0)
        i2 = min(i0 + self.hw * 3, nx)
        # img_raw = img0[j1:j2, i1:i2].copy()

        # masking noisy background
        maxv = np.max(np.max(img1))
        inds_mask = np.asarray(img1 < maxv * self.noise_thresh).nonzero()
        img0[inds_mask] = 0.0

        # remove very high isolated values
        img1 = ndimage.uniform_filter(img0, size=3)
        inds_hi = np.asarray(1.5 * img1 < img0).nonzero()
        img0[inds_hi] = img1[inds_hi]

        img = img0[j1:j2, i1:i2]

        image = self.shearing(img)
        if self.fliplr:
            image = np.fliplr(image)
        return image

    def process(self):
        """
            function processes TDS images

            :param filename: path to a pickled image file
            :param bg_image: background image
            :param dx_px: ps/px, pixel size in horizontal direction
            :param dy_px: MeV/px, pixel size in vertical direction
            :param charge: pC
            :param noise_thresh: 0.1, remove background with value < noise_thresh * max(image)
            :param plotting: False
            :return:
            """
        if self.raw_image is None:
            if not self.read_image_from_pickle():
                print("No image")
                return
        if self.raw_image.__class__ is list:
            if len(self.raw_image) > 1:
                images = []
                for img in self.raw_image:
                    images.append(self.process_image(img))

            else:
                images = [self.process_image(self.raw_image[0])]
        else:
            images = [self.process_image(self.raw_image)]
        self.images = self.center_images(images)
        img = self.averaging_images(self.images)
        # gauss filter
        self.image = ndimage.gaussian_filter(img, sigma=self.gauss_filter_sigma)


    def get_slice_parameters(self, image):
        (j0f, i0f) = ndimage.center_of_mass(image)
        # i0i = int(i0f)
        # j0i = int(j0f)
        (ny, nx) = image.shape

        # energy axis
        y = (np.linspace(0, ny - 1, ny) - j0f) * self.dy_px
        # time axis
        x = (np.linspace(0, nx - 1, nx) - i0f) * self.dx_px

        ro = sum(image)
        current = self.charge / self.dx_px * ro / sum(ro)

        energy_proj = sum(image.T)
        energy_proj = self.charge / self.dy_px * energy_proj / sum(energy_proj)

        m = sum(x * current) / sum(current)
        m2 = sum((x - m) ** 2 * current) / sum(current)
        x_sigma = np.sqrt(m2)
        slice_energy = np.sum(image.T * y, axis=1) * self.dy_px

        S = np.sum(image, axis=0) * self.dy_px
        # print("S = ", S)
        inds_roi = np.asarray(S > self.noise_proj_thresh * np.max(S)).nonzero()[0]
        slice_energy[inds_roi] = slice_energy[inds_roi] / S[inds_roi]
        slice_energy_spread = np.zeros(nx)

        for j in range(len(inds_roi)):
            i = inds_roi[j]
            y2 = (y - slice_energy[i]) ** 2
            y_sigma0 = np.sum(image[:, i] * y2) * self.dy_px
            slice_energy_spread[i] = np.sqrt(y_sigma0 / S[i])

        A, mu, sigma_x_fit = self.gauss_fit(x, current, x_sigma)
        return x, y, current, energy_proj, slice_energy, slice_energy_spread, np.abs(sigma_x_fit), A, inds_roi

    def gauss_fit(self, x, y, sigma_x_estm=None):
        if sigma_x_estm is None:
            sigma_x_estm = (np.max(x) - np.min(x))/4.

        def gauss(x, *p):
            A, mu, sigma = p
            return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        p0 = [np.max(y), 0., sigma_x_estm]

        coeff, var_matrix = curve_fit(gauss, x, y, p0=p0)
        A, mu, sigma = coeff
        return A, mu, sigma

    def plot(self, title=None, figsize=(11,9)):
        if self.image is None:
            self.process()

        x, y, current, energy_proj, slice_energy, slice_energy_spread, x_sigma_fit, A_fit, inds_roi = self.get_slice_parameters(self.image)
        fig = plt.figure(10, figsize=figsize)
        st = fig.suptitle(title, fontsize="x-large")

        ax_im = plt.subplot(221)
        (ny, nx) = self.image.shape
        my_rainbow = copy.deepcopy(plt.get_cmap('rainbow'))
        my_rainbow.set_under('w')
        vmin = np.min(self.image) + (np.max(self.image) - np.min(self.image)) * 0.01
        # plt.imshow(self.img, extent=[self.x[0], self.x[nx - 1], self.y[0], self.y[ny - 1]],
        #            aspect='auto', origin='lower')

        ax_im.imshow(self.image, extent=[x[0], x[nx - 1], y[0], y[ny - 1]],
                     aspect='auto', origin='lower', vmin=vmin, cmap=my_rainbow)
        plt.ylabel("dE [MeV]")
        ax_im.plot(x[inds_roi], slice_energy[inds_roi], 'r:')
        #plt.plot(self.x[self.inds_roi], self.slice_energy[self.inds_roi] + 3 * self.slice_energy_spread[self.inds_roi], 'w:',
        #         self.x[self.inds_roi], self.slice_energy[self.inds_roi] - 3 * self.slice_energy_spread[self.inds_roi], 'w:')
        plt.setp(ax_im.get_xticklabels(), visible=False)


        plt.title("processed data")


        ax_cu = plt.subplot(223, sharex=ax_im)
        ax_cu.plot(x, current)


        mu=0
        ax_cu.set_xlabel('t [ps]')
        ax_cu.set_xlim(mu - 5*np.abs(x_sigma_fit), mu + 5 * np.abs(x_sigma_fit))
        ax_cu.set_ylabel('I [A]')
        ax_cu.set_title('current, fitted gauss rms=' + "{:.2f}".format(x_sigma_fit) + ' ps')

        ax_ep = plt.subplot(222, sharey=ax_im)

        plt.plot(energy_proj, y)#, transform= rot + base)
        yl_max = np.max(slice_energy[inds_roi] + 6 * slice_energy_spread[inds_roi])
        yl_min = np.min(slice_energy[inds_roi] - 6 * slice_energy_spread[inds_roi])
        ax_ep.set_ylim(yl_min, yl_max)
        plt.setp(ax_ep.get_yticklabels(), visible=False)
        plt.setp(ax_ep.get_xticklabels(), visible=False)

        ax_ep.xaxis.set_label_position("top")
        plt.title('Projection on the energy axis')

        ax_es = plt.subplot(224, sharex=ax_im)
        ax_es.yaxis.tick_right()
        ax_es.yaxis.set_label_position("right")
        plt.plot(x[inds_roi], slice_energy_spread[inds_roi] * 1e3)
        plt.xlabel('t [ps]')
        plt.ylabel('$\sigma_E$, [keV]')
        plt.title('slice energy spread')
        plt.show()

    def plot_all(self, title=None, figsize=(9, 7)):
        if self.images is None:
            self.process()

        fig = plt.figure(10, figsize=figsize)
        st = fig.suptitle(title, fontsize="x-large")

        ax_im = plt.subplot(221)

        my_rainbow = copy.deepcopy(plt.get_cmap('rainbow'))
        my_rainbow.set_under('w')
        vmin = np.min(self.image) + (np.max(self.image) - np.min(self.image)) * 0.02
        plt.ylabel("dE [MeV]")
        plt.setp(ax_im.get_xticklabels(), visible=False)
        plt.title("processed data")

        ax_cu = plt.subplot(223, sharex=ax_im)
        ax_cu.set_xlabel('t [ps]')
        ax_cu.set_ylabel('I [A]')

        ax_ep = plt.subplot(222, sharey=ax_im)
        plt.setp(ax_ep.get_yticklabels(), visible=False)
        plt.setp(ax_ep.get_xticklabels(), visible=False)
        ax_ep.xaxis.set_label_position("top")
        ax_ep.set_title('Projection on the energy axis')

        ax_es = plt.subplot(224, sharex=ax_im)
        ax_es.yaxis.tick_right()
        ax_es.yaxis.set_label_position("right")
        ax_es.set_xlabel('t [ps]')
        ax_es.set_ylabel('$\sigma_E$, [keV]')
        ax_es.set_title('slice energy spread')

        for i, img in enumerate(self.images):
            x, y, current, energy_proj, slice_energy, slice_energy_spread, x_sigma_fit, A_fit, inds_roi = self.get_slice_parameters(img)

            if i == 0:
                (ny, nx) = img.shape
                # plt.imshow(self.img, extent=[self.x[0], self.x[nx - 1], self.y[0], self.y[ny - 1]],
                #            aspect='auto', origin='lower')

                ax_im.imshow(self.images[0], extent=[x[0], x[nx - 1], y[0], y[ny - 1]],
                             aspect='auto',
                             origin='lower',
                             #vmin=vmin, cmap=my_rainbow
                             )
                ax_im.plot(x[inds_roi], slice_energy[inds_roi], 'r:')
                #plt.plot(self.x[self.inds_roi], self.slice_energy[self.inds_roi] + 3 * self.slice_energy_spread[self.inds_roi], 'w:',
                #         self.x[self.inds_roi], self.slice_energy[self.inds_roi] - 3 * self.slice_energy_spread[self.inds_roi], 'w:')

                mu = 0.
                ax_cu.set_xlim(mu - 5*np.abs(x_sigma_fit), mu + 5 * np.abs(x_sigma_fit))

                ax_cu.set_title('current, fitted gauss rms=' + "{:.2f}".format(x_sigma_fit) + ' ps')

                yl_max = np.max(slice_energy[inds_roi] + 6 * slice_energy_spread[inds_roi])
                yl_min = np.min(slice_energy[inds_roi] - 6 * slice_energy_spread[inds_roi])
                ax_ep.set_ylim(yl_min, yl_max)


            ax_cu.plot(x, current)
            ax_ep.plot(energy_proj, y)#, transform= rot + base)

            ax_es.plot(x[inds_roi], slice_energy_spread[inds_roi] * 1e3)

        plt.show()


def process_tds_image(filename, bg_image, dx_px, dy_px, charge, noise_thresh=0.1, plotting=False):
    """
    function processes TDS images

    :param filename: path to a pickled image file
    :param bg_image: background image
    :param dx_px: ps/px, pixel size in horizontal direction
    :param dy_px: MeV/px, pixel size in vertical direction
    :param charge: pC
    :param noise_thresh: 0.1, remove background with value < noise_thresh * max(image)
    :param plotting: False
    :return:
    """
    # read image
    with open(filename, 'rb') as f:
        img0 = pickle.load(f)

    img0 = img0.astype(np.double) - bg_image
    img0 = np.flipud(img0)
    img1 = ndimage.uniform_filter(img0, size=150)
    (ny, nx) = img0.shape

    # clip image
    (j0,i0) = np.unravel_index(np.argmax(img1, axis=None), img1.shape)
    hw = 100
    j1 = max(j0 - hw*5, 0)
    j2 = min(j0 + hw*5, ny)
    i1 = max(i0 - hw*3, 0)
    i2 = min(i0 + hw*3, nx)
    img_raw = img0[j1:j2, i1:i2].copy()

    # masking noisy background
    maxv = np.max(np.max(img1))
    inds = np.asarray(img1 < maxv * noise_thresh).nonzero()
    img0[inds] = 0.0

    # remove very high isolated values
    img1 = ndimage.uniform_filter(img0, size=3)
    inds = np.asarray(1.5*img1 < img0).nonzero()
    img0[inds] = img1[inds]
    img = ndimage.gaussian_filter(img0[j1:j2, i1:i2], sigma=0)
    (j0f, i0f) = ndimage.center_of_mass(img)
    i0i = int(i0f)
    j0i = int(j0f)
    (ny, nx) = img.shape

    # energy axis
    y = (np.linspace(0, ny-1, ny) - j0f) * dy_px
    # time axis
    x = (np.linspace(0, nx-1, nx) - i0f) * dx_px

    ro = sum(img)
    I = charge / dx_px * ro / sum(ro)

    roE = sum(img.T)
    roE = charge / dy_px * roE / sum(roE)

    m = sum(x * I) / sum(I)
    m2 = sum((x - m) ** 2 * I) / sum(I)
    sigx = np.sqrt(m2)
    dE_m = np.sum(img.T * y, axis=1) * dy_px

    S = np.sum(img, axis=0) * dy_px

    inds = np.asarray(S > 0.1 * np.max(S)).nonzero()[0]
    dE_m[inds] = dE_m[inds] / S[inds]
    dE_sigma = np.zeros(nx)
    # plt.plot(dE_m)
    # plt.show()

    for j in range(len(inds)):
        i = inds[j]
        y2 = (y - dE_m[i])**2
        y_sigma0 = np.sum(img[:, i]*y2) * dy_px
        dE_sigma[i] = np.sqrt(y_sigma0 / S[i])



    if plotting:
        X, Y = np.meshgrid(x, y)
        fig = plt.figure(10)
        ax3 = fig.add_subplot(2, 1, 1, projection='3d')
        ax3.plot_wireframe(X, Y, img_raw)
        plt.title("raw data")
        plt.xlabel("t[fs]")
        plt.ylabel("dE[MeV ]")
        ax4 = fig.add_subplot(2, 1, 2)
        ax4.imshow(img_raw, extent=[x[0], x[nx - 1], y[0], y[ny - 1]], aspect='auto', origin='lower')
        ax4.set(xlabel="t[ps]", ylabel="dE[MeV]")
        plt.xlabel("t[fs]")
        plt.ylabel("dE[MeV ]")

        fig = plt.figure(20)
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.plot_wireframe(X, Y, img)
        plt.title("processed data")
        plt.xlabel("t[ps]")
        plt.ylabel("dE[MeV ]")

        fig = plt.figure(21)
        plt.imshow(img, extent=[x[0], x[nx - 1], y[0], y[ny - 1]], aspect='auto', origin='lower')
        plt.plot(x[i0i], y[j0i], 'wo')
        plt.ylabel("dE[MeV]")
        plt.plot(x[inds], dE_m[inds], 'r:')
        plt.plot(x[inds], dE_m[inds] + 3 * dE_sigma[inds], 'w:', x[inds], dE_m[inds] - 3 * dE_sigma[inds], 'w:')
        plt.xlabel('t[ps]')
        plt.ylabel('dE[keV]')
        plt.title("processed data")

        fig = plt.figure(30)
        plt.plot(x, I)
        plt.xlabel('t[ps]')
        plt.ylabel('I[A]')
        plt.title('current, rms=' + "{:.2f}".format(sigx) + ' fs')

        fig = plt.figure(31)
        plt.plot(y, roE)
        plt.xlabel('dE[MeV]')
        plt.ylabel('charge density[a.u.]')

        fig=plt.figure(32)
        plt.plot(x[inds], dE_sigma[inds]*1e3)
        plt.xlabel('t[ps]')
        plt.ylabel('dE[keV]')
        plt.title('slice energy spread')
        plt.show()
    return img, x, y, sigx, I, roE, dE_m, dE_sigma, inds


def extract_parameters(image_dict, parent_dir, tds_img, show_img=False):
    l2_chirps = list(image_dict.keys())
    img_paths = [image_dict[chirp]["raw"] for chirp in l2_chirps]
    data_extract = {}
    for i, key  in enumerate(image_dict):

        filenames = [str(parent_dir) + "/" + img_path[2:-3] + "pcl" for img_path in img_paths[i]]


        tds_img.raw_image = tds_img.load_images(filenames)
        tds_img.process()
        if show_img:
            tds_img.plot_all(title="L1 chirp: " + str(np.round(l2_chirps[i], 2)), figsize=(11, 9))
            plt.show()
        sigmas = []
        a = []
        c_max = []
        for i, img in enumerate(tds_img.images):
            x, y, current, energy_proj, slice_energy, slice_energy_spread, x_sigma_fit, A_fit, inds_roi = tds_img.get_slice_parameters(
                img)
            sigmas.append(x_sigma_fit)
            a.append(A_fit)
            c_max.append(np.max(current))


        data_extract[key] = {"sigma_mean":np.mean(sigmas), "sigma_std":np.std(sigmas),
                                      "fit_ampl_mean":np.mean(a), "fit_ampl_std":np.std(a),
                                      "ampl_mean": np.mean(c_max), "ampl_std": np.std(c_max)}
    return data_extract
