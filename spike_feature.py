import os
import numpy as np
from PIL import Image
import csv
from skimage.io import imread
from skimage.measure import label, regionprops
from scipy.spatial.distance import pdist, squareform
from skimage.color import rgb2hsv


# ---------------------- 核心工具函数 ---------------------- #
def is_edge_spike(region, img_width, img_height):
    y_min, x_min, y_max, x_max = region.bbox
    return (y_min == 0) or (y_max == img_height) or (x_min == 0) or (x_max == img_width)


def filter_edge_spikes(regions, img_width, img_height, exclude_edge=True):
    if not exclude_edge:
        return regions
    return [r for r in regions if not is_edge_spike(r, img_width, img_height)]


def calculate_veg_indices(rgb_img, mask_img):
    rgb_norm = rgb_img / 255.0
    R, G, B = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

    ExG  = 2*G - R - B
    NGRDI = (G - R) / (G + R +1e-8)
    GLI  = (2*G - R - B) / (2*G + R + B + 1e-8)
    EXR  = 1.4*R-G
    TGI  = -0.5 * (190*(R - G) - 120*(R - B))
    ExY = R + G - 2*B
    YEI = (R * G) / (B + 1e-8)
    RGI = R / (G + 1e-8)
    indices = {
        "ExG": ExG, "NGRDI": NGRDI, "GLI": GLI, "ExR": EXR, "TGI": TGI,
        "ExY": ExY, "YEI": YEI, "RGI": RGI
    }

    def stats(arr, mask):
        values = arr
        if values.size == 0:
            return (0.000, 0.000, 0.000)
        mean = np.mean(values)
        var  = np.var(values)
        cv   = np.std(values) / mean if mean != 0 else 0
        return (np.round(mean, 3), np.round(var, 3), np.round(cv, 3))

    return {k: stats(v, mask_img) for k, v in indices.items()}


def calculate_morph_features(filtered_regions):
    spike_count = len(filtered_regions)
    if spike_count == 0:
        return spike_count, 0.000, 0.000, 0.000
    areas = [r.area for r in filtered_regions]
    total_area = np.round(sum(areas), 3)
    avg_area = np.round(np.mean(areas), 3)
    perimeters = [r.perimeter for r in filtered_regions if r.perimeter > 0]
    avg_perimeter = np.round(np.mean(perimeters) if perimeters else 0, 3)
    return spike_count, avg_area, total_area, avg_perimeter

def calculate_color_variation(rgb_img, filtered_regions):
    if not filtered_regions:
        return 0.000
    h_means, s_means, v_means = [], [], []
    for r in filtered_regions:
        coords = r.coords
        pixels = rgb_img[coords[:, 0], coords[:, 1]] / 255.0
        if len(pixels) == 0:
            continue
        pixels_hsv = rgb2hsv(pixels)
        h_means.append(np.mean(pixels_hsv[:, 0]))
        s_means.append(np.mean(pixels_hsv[:, 1]))
        v_means.append(np.mean(pixels_hsv[:, 2]))
    if not h_means:
        return 0.000
    h_cv = np.std(h_means) / np.mean(h_means) if np.mean(h_means) != 0 else 0
    s_cv = np.std(s_means) / np.mean(s_means) if np.mean(s_means) != 0 else 0
    v_cv = np.std(v_means) / np.mean(v_means) if np.mean(v_means) != 0 else 0
    color_cv = np.sqrt(h_cv**2 + s_cv**2 + v_cv**2)
    return np.round(color_cv, 3)


def mean_nearest_neighbor(all_regions):
    if len(all_regions) < 2:
        return 0.000
    centroids = np.array([[r.centroid[1], r.centroid[0]] for r in all_regions])
    dist_matrix = squareform(pdist(centroids))
    np.fill_diagonal(dist_matrix, np.inf)
    min_distances = np.min(dist_matrix, axis=1)
    return np.round(np.mean(min_distances), 3)


def calculate_nni(all_regions, img_width, img_height):
    if len(all_regions) < 2:
        return 0.000
    total_area = img_width * img_height
    total_spike = len(all_regions)
    rho = total_spike / total_area
    if rho == 0:
        return 0.000
    d_expected = 1 / (2 * np.sqrt(rho))
    d_observed = mean_nearest_neighbor(all_regions)
    return np.round(d_observed / d_expected, 3)


def spatial_entropy(all_regions, img_width, img_height, grid_size=5):
    if len(all_regions) == 0:
        return 0.000
    centroids = np.array([[r.centroid[1], r.centroid[0]] for r in all_regions])
    x, y = centroids[:, 0], centroids[:, 1]
    x_bins = np.linspace(0, img_width, grid_size + 1)
    y_bins = np.linspace(0, img_height, grid_size + 1)
    grid_counts, _, _ = np.histogram2d(y, x, bins=[y_bins, x_bins])
    grid_counts = grid_counts.flatten()
    valid_counts = grid_counts[grid_counts > 0]
    probs = valid_counts / len(all_regions)
    entropy = -np.sum(probs * np.log2(probs))
    return np.round(entropy, 3)


def density_cv(all_regions, img_width, img_height):
    if len(all_regions) < 2:
        return 0.000
    centroids = np.array([[r.centroid[1], r.centroid[0]] for r in all_regions])
    x, y = centroids[:, 0], centroids[:, 1]
    from scipy.stats import gaussian_kde
    kde = gaussian_kde([x, y])
    xx, yy = np.mgrid[0:img_width:10j, 0:img_height:10j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)
    density_values = density.flatten()
    density_values = density_values[density_values > np.percentile(density_values, 10)]
    if len(density_values) < 2:
        return 0.000
    mean_density = np.mean(density_values)
    std_density = np.std(density_values)
    cv = (std_density / mean_density) * 100 if mean_density != 0 else 0
    return np.round(cv, 3)


def morans_i(all_regions):
    if len(all_regions) < 2:
        return 0.000
    centroids = np.array([[r.centroid[1], r.centroid[0]] for r in all_regions])
    n = len(all_regions)
    dist_matrix = squareform(pdist(centroids))
    np.fill_diagonal(dist_matrix, 0)
    weight_matrix = 1 / (dist_matrix + 1e-8)
    x_coords = centroids[:, 0]
    x_mean = np.mean(x_coords)
    dev_x = x_coords - x_mean
    numerator = np.sum(weight_matrix * dev_x.reshape(-1, 1) * dev_x.reshape(1, -1))
    denominator = np.sum(weight_matrix) * np.var(x_coords)
    mi = (n / denominator) * numerator if denominator != 0 else 0
    return np.round(mi, 4)


# ---------------------- 主流程 ---------------------- #
if __name__ == "__main__":
    mask_folder = r"hhh_photos/spike_predict/segmented"
    rgb_folder = r"hhh_photos/spike_resize"
    csv_path = r"spike_features.csv"

    exclude_edge_spikes = True

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "spike_count", "avg_spike_area", "total_area", "avg_perimeter",
            "spatial_density", "nni", "spatial_entropy", "density_cv", "morans_i",
            "ExG_mean", "ExG_var", "ExG_cv",
            "NGRDI_mean", "NGRDI_var", "NGRDI_cv",
            "GLI_mean", "GLI_var", "GLI_cv",
            "ExR_mean", "ExR_var", "ExR_cv",
            "TGI_mean", "TGI_var", "TGI_cv",
            "ExY_mean", "ExY_var", "ExY_cv",
            "YEI_mean", "YEI_var", "YEI_cv",
            "RGI_mean", "RGI_var", "RGI_cv",
        ])

        for mask_file in os.listdir(mask_folder):
            if "spike_1" not in mask_file:
                continue
            if not mask_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            mask_file_name = os.path.splitext(mask_file)[0]
            rgb_file = os.path.join(rgb_folder, f"{mask_file_name}.jpg")
            if not os.path.exists(rgb_file):
                print(f"警告：未找到RGB文件 {mask_file}")
                continue

            try:
                mask_path = os.path.join(mask_folder, mask_file)
                mask_img = imread(mask_path)
                if mask_img.ndim > 2:
                    mask_img = np.mean(mask_img, axis=2).astype(np.uint8)

                unique_values = np.unique(mask_img)
                unique_values = unique_values[unique_values > 0]
                all_regions = []
                for val in unique_values:
                    binary_mask = (mask_img == val).astype(np.uint8)
                    labeled = label(binary_mask)
                    props = regionprops(labeled)
                    if props:
                        all_regions.append(props[0])

                total_spike = len(all_regions)

                img = Image.open(mask_path)
                img_width, img_height = img.size
                filtered_regions = filter_edge_spikes(all_regions, img_width, img_height, exclude_edge_spikes)

                spike_count, avg_spike_area, total_area, avg_perimeter = calculate_morph_features(filtered_regions)

                rgb_img = imread(rgb_file)
                color_variation = calculate_color_variation(rgb_img, filtered_regions)

                mask_binary = (mask_img > 0).astype(np.uint8)
                veg_indices = calculate_veg_indices(rgb_img, mask_binary)

                if total_spike == 0:
                    spatial_density = 0
                    nni = 0
                    spatial_ent = 0
                    dens_cv = 0
                    mi = 0
                else:
                    spatial_density = (total_spike / (img_width * img_height)) * 1e6
                    nni = calculate_nni(all_regions, img_width, img_height)
                    spatial_ent = spatial_entropy(all_regions, img_width, img_height)
                    dens_cv = density_cv(all_regions, img_width, img_height)
                    mi = morans_i(all_regions)

                writer.writerow([
                    mask_file,
                    total_spike, avg_spike_area, total_area, avg_perimeter,
                    spatial_density, nni, spatial_ent, dens_cv, mi,
                    veg_indices["ExG"][0], veg_indices["ExG"][1], veg_indices["ExG"][2],
                    veg_indices["NGRDI"][0], veg_indices["NGRDI"][1], veg_indices["NGRDI"][2],
                    veg_indices["GLI"][0], veg_indices["GLI"][1], veg_indices["GLI"][2],
                    veg_indices["ExR"][0], veg_indices["ExR"][1], veg_indices["ExR"][2],
                    veg_indices["TGI"][0], veg_indices["TGI"][1], veg_indices["TGI"][2],
                    veg_indices["ExY"][0], veg_indices["ExY"][1], veg_indices["ExY"][2],
                    veg_indices["YEI"][0], veg_indices["YEI"][1], veg_indices["YEI"][2],
                    veg_indices["RGI"][0], veg_indices["RGI"][1], veg_indices["RGI"][2],
                ])
                print(f"已处理：{mask_file}")

            except Exception as e:
                print(f"处理失败：{mask_file} - {str(e)}")
                writer.writerow([mask_file] + [None] * 27)

    print(f"分析完成！结果已保存至：{csv_path}")
