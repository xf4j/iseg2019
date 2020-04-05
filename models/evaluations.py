import numpy as np
from scipy.ndimage import morphology

def compute_metric(gt, model, voxel_spacing):
    dice = np.sum(gt * model) / (np.sum(gt) + np.sum(model)) * 2.0
    msd, hd, hd95 = calculate_distance(gt, model, voxel_spacing)
    # hd is often not used
    return dice, msd, hd95

'''
# Only works for 2D
def get_points(mask, spacing):
    points = []
    if np.amax(mask) > 0:
        contours = measure.find_contours(mask.astype(np.float32), 0.5)
        for contour in contours:
            con = contour.ravel().reshape((-1, 2)) # con[0] is along x (last dimension of mask)
            for p in con:
                points.append([p[1] * spacing[0], p[0] * spacing[1]])
    return np.asarray(points, dtype=np.float32)
    
def calculate_distance(mask1, mask2, voxel_spacing):
    first_slice = True
    for z in range(mask1.shape[0]):
        points1 = get_points(mask1[z], voxel_spacing[1 : 3])
        points2 = get_points(mask2[z], voxel_spacing[1 : 3])
        if points1.shape[0] > 0 and points2.shape[0] > 0:
            dists = spatial.distance.cdist(points1, points2, metric='euclidean')
            if first_slice:
                dist12 = np.amin(dists, axis=1)
                dist21 = np.amin(dists, axis=0)
                first_slice = False
            else:
                dist12 = np.append(dist12, np.amin(dists, axis=1))
                dist21 = np.append(dist21, np.amin(dists, axis=0))
    # Mean surface distance
    msd = (np.mean(dist12) + np.mean(dist21)) / 2.0
    # Hausdorff distance
    hd = (np.amax(dist12) + np.amax(dist21)) / 2.0
    # 95 Hausdorff distance
    hd95 = (np.percentile(dist12, 95) + np.percentile(dist21, 95)) / 2.0
    return msd, hd95
'''

# https://mlnotebook.github.io/post/surface-distance-function/
def calculate_distance(input1, input2, spacing, connectivity=1):
    conn = morphology.generate_binary_structure(input1.ndim, connectivity)
    
    s1 = input1 - morphology.binary_erosion(input1, conn)
    s2 = input2 - morphology.binary_erosion(input2, conn)
    
    dta = morphology.distance_transform_edt(1 - s1, spacing)
    dtb = morphology.distance_transform_edt(1 - s2, spacing)
    
    # distance from surface 1 to 2
    dist12 = np.ravel(dtb[s1 != 0])
    dist21 = np.ravel(dta[s2 != 0])
    
    # Mean surface distance
    msd = (np.mean(dist12) + np.mean(dist21)) / 2.0
    # Hausdorff distance
    hd = (np.amax(dist12) + np.amax(dist21)) / 2.0
    # 95 Hausdorff distance
    hd95 = (np.percentile(dist12, 95) + np.percentile(dist21, 95)) / 2.0
    
    return msd, hd, hd95