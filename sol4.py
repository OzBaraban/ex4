import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter, convolve
from scipy.ndimage import label, center_of_mass
from scipy.ndimage import map_coordinates
import shutil
from imageio import imwrite
import sol4_utils
from random import randrange


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    i_x = convolve(im, np.array([[1, 0, -1]]))
    i_y = convolve(im, np.array([[1], [0], [-1]]))
    i_x_2 = sol4_utils.blur_spatial(i_x * i_x, 3)
    i_y_2 = sol4_utils.blur_spatial(i_y * i_y, 3)
    i_x_y = sol4_utils.blur_spatial(i_x * i_y, 3)
    i_y_x = sol4_utils.blur_spatial(i_y * i_x, 3)
    # determinant
    det = (i_x_2 * i_y_2) - (i_x_y * i_y_x)
    # trace
    trace = i_x_2 + i_y_2
    harris_response = det - 0.04 * (trace * trace)
    r_after_nms = np.transpose(non_maximum_suppression(harris_response))
    return np.argwhere(r_after_nms == 1)


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    win_size = desc_rad * 2 + 1
    n = pos.shape[0]
    # creates n windows at the wanted size (n,n) for each point
    window = np.zeros((n, win_size, win_size))
    # normalizing the coordinates of corners according to a 3rd level pyramid
    leveled_pyr_pos = pos / 4
    for i in range(len(leveled_pyr_pos)):
        # creating the x and y ranges for the descriptor
        x_range = np.arange(leveled_pyr_pos[i, 0] - desc_rad, leveled_pyr_pos[i, 0] + desc_rad + 1)
        y_range = np.arange(leveled_pyr_pos[i, 1] - desc_rad, leveled_pyr_pos[i, 1] + desc_rad + 1)
        xx, yy = np.meshgrid(x_range, y_range)
        # interpolating this level with the one above
        interpolate = map_coordinates(im, [yy, xx], order=1, prefilter=False)
        mean = np.mean(interpolate)
        norm = np.linalg.norm(interpolate - mean)
        # normalizing the function
        if norm == 0:
            window[i, :, :] = 0
        else:
            window[i, :, :] = (interpolate - mean) / norm
    return window


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
             1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
             2) A feature descriptor array with shape (N,K,K)
    """
    first_level = pyr[0]
    third_level = pyr[2]
    corners_list = spread_out_corners(first_level, 7, 7, 16)
    descriptors = sample_descriptor(third_level, corners_list, 3)
    return [corners_list, descriptors]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
               1) An array with shape (M,) and dtype int of matching indices in desc1.
               2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    # getting the parameters
    n1, n2 = desc1.shape[0], desc2.shape[0]
    k = desc1.shape[1]
    # transforming from 3d vectors to 2d
    desc1_2d = desc1.reshape(n1, k ** 2)
    desc2_2d = desc2.reshape(n2, k ** 2)
    # transpose because of dot product
    desc2_2d = desc2_2d.T
    # calculate s
    s = np.dot(desc1_2d, desc2_2d)
    # filtering the rows and cols
    row_filter = np.partition(s, -2, 0)[-2, :]
    col_filter = np.partition(s, -2, 1)[:, -2]
    # apply the conditions on the s as shown in the targil
    s[s <= min_score] = 0
    s[s < row_filter] = 0
    s[(s.T < col_filter.T).T] = 0
    return np.nonzero(s)


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogeneous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12
    """
    # adding 1 as the third coordinate
    n = pos1.shape[0]
    col_2_add = np.ones(n)
    homo_axis = np.column_stack((pos1, col_2_add))
    # apply the matrix
    pos_shifted = (np.dot(H12, homo_axis.T)).T
    z_2_divide = pos_shifted[:, 2]
    pos_shifted_without_z = np.delete(pos_shifted, 2, 1)
    pos_x = pos_shifted_without_z[:, 0] / z_2_divide
    pos_y = pos_shifted_without_z[:, 1] / z_2_divide
    xy = np.column_stack((pos_x, pos_y))
    return xy


def ransac_homography(pos1, pos2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                  1) A 3x3 normalized homography matrix.
                  2) An Array with shape (S,) where S is the number of inliers,
                      containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    max_inliners = 0
    i = 0
    while i < num_iter:
        n = pos1.shape[0]
        # pick 2 random points from the image
        idx = randrange(n - 1)
        if translation_only is False:
            p1j = np.array([pos1[idx, :], pos1[idx + 1, :]])
            p2j = np.array([pos2[idx, :], pos2[idx + 1, :]])
        else:
            p1j = np.array([pos1[idx, :]])
            p2j = np.array([pos2[idx, :]])
        h12 = estimate_rigid_transform(p1j, p2j, translation_only)
        p2jtag = apply_homography(pos1, h12)
        diff = (p2jtag - pos2) ** 2
        # create a distance matrix
        distance_mat = np.sum(diff, axis=1)
        # index of inliers
        inlier = np.argwhere(distance_mat < inlier_tol)
        # finding the number of inliers
        num_of_inliers = inlier.shape[0]
        if num_of_inliers > max_inliners:
            inlier_final = inlier
            max_inliners = num_of_inliers
            h = h12
        i += 1
    h /= h[2, 2]
    inlier_final = inlier_final.flatten()
    return h, np.array(inlier_final)


def display_matches(im1, im2, points1, points2, inliers):
    """
    Display matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :param points1: An array shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An array shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    p2_cor_start = im1.shape[1]
    # display both images as grayscale
    both_images = np.hstack((im1, im2))
    plt.imshow(both_images, cmap="gray")
    filter_mat = np.ones(points1.shape, bool)
    # an index of an inliers is marked as False in the Filter
    filter_mat[inliers] = False
    # marking the points which are outliers
    p1_outliers = points1[filter_mat]
    p2_outliers = points2[filter_mat]
    # reshaping them to the new size - without the inliers
    p1_outliers = p1_outliers.reshape(points1.shape[0] - inliers.size, 2)
    p2_outliers = p2_outliers.reshape(points1.shape[0] - inliers.size, 2)
    x_outliers = np.vstack((p1_outliers[:, 0], p2_outliers[:, 0] + p2_cor_start))
    y_outliers = np.vstack((p1_outliers[:, 1], p2_outliers[:, 1]))
    # marking the points which are inliers
    x_inliers = np.vstack((points1[inliers, 0], points2[inliers, 0] + p2_cor_start))
    y_inliers = np.vstack((points1[inliers, 1], points2[inliers, 1]))
    # plotting the outliers and inliers
    plt.plot(x_outliers, y_outliers, color='blue', linestyle='solid', marker='o', markerfacecolor='red', markersize=5,
             linewidth=1)
    plt.plot(x_inliers, y_inliers, color='yellow', linestyle='solid', marker='o', markerfacecolor='red', markersize=5,
             linewidth=1)
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a list of homographies to a common reference frame.
    :param H_succesive: A list of M-1 3x3 homography matrices where H_successive[i] is a homography which transforms
    points from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to accumulate the given homographies.
    :return: A list of M 3x3 homography matrices, where H2m[i] transforms points from coordinate system i to
    coordinate system m
    """
    M = len(H_succesive)
    h_tot = [np.eye(3)]*(M+1)
    for i in range(m-1, -1, -1):
        h_curr = np.dot(h_tot[i+1], H_succesive[i])
        h_curr /= h_curr[2][2]
        h_tot[i] = h_curr
    for i in range(m+1, M):
        h_curr = np.dot(h_tot[i-1], np.linalg.inv(H_succesive[i-1]))
        h_curr /= h_curr[2][2]
        h_tot[i] = h_curr
    return h_tot


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner, and the second row is the [x,y] of the
    bottom right corner
    """
    top_left = np.array([0, 0])
    bottom_right = np.array([w-1, h-1])
    top_right = np.array([w-1, 0])
    bottom_left = np.array([0, h-1])
    pos1 = np.array((top_left, top_right, bottom_left, bottom_right))
    xy_homo = apply_homography(pos1, homography)
    min_x = np.min(xy_homo[:, 0])
    min_y = np.min(xy_homo[:, 1])
    max_x = np.max(xy_homo[:, 0])
    max_y = np.max(xy_homo[:, 1])
    corners = np.array([[min_x, min_y], [max_x, max_y]], dtype=int)
    return corners


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homography.
    :return: A 2d warped image.
    """
    h, w = image.shape
    corners_2_image = compute_bounding_box(homography, w, h)
    top_left, bottom_right = corners_2_image[0], corners_2_image[1]
    h_range = np.arange(top_left[0], bottom_right[0]+1)
    w_range = np.arange(top_left[1], bottom_right[1]+1)
    xx, yy = np.meshgrid(h_range, w_range)
    xy_in_box = np.dstack([xx, yy])
    xy_in_box = np.reshape(xy_in_box, (-1, 2))
    h_inverse = np.linalg.inv(homography)
    x_tag_y_tag = apply_homography(xy_in_box, h_inverse)
    x_tag, y_tag = x_tag_y_tag[:, 0], x_tag_y_tag[:, 1]
    interpolate = map_coordinates(image, [y_tag, x_tag], order=1, prefilter=False)
    return interpolate.reshape(len(w_range), len(h_range))


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homography.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maxima.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False
    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True
    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
