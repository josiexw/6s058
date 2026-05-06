import cv2
import numpy as np


def create_sgbm_matcher(
    window_size: int = 7,
    num_disp: int = 96,
    uniqueness_ratio: int = 10,
    speckle_window_size: int = 150,
    speckle_range: int = 2,
):
    if window_size % 2 == 0:
        window_size += 1

    if num_disp % 16 != 0:
        num_disp = ((num_disp // 16) + 1) * 16

    p1 = 8 * 3 * window_size ** 2
    p2 = 32 * 3 * window_size ** 2

    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    return matcher


def _prepare_sgbm_input(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def run_sgbm(left_gray: np.ndarray, right_gray: np.ndarray, matcher=None) -> np.ndarray:
    if matcher is None:
        matcher = create_sgbm_matcher()

    left_gray = _prepare_sgbm_input(left_gray)
    right_gray = _prepare_sgbm_input(right_gray)

    disp = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan

    return disp


def apply_wls_filter(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_matcher,
    lambda_value: float = 8000,
    sigma_color: float = 1.5,
) -> np.ndarray:
    left_gray = _prepare_sgbm_input(left_gray)
    right_gray = _prepare_sgbm_input(right_gray)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    disp_left = left_matcher.compute(left_gray, right_gray)
    disp_right = right_matcher.compute(right_gray, left_gray)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(lambda_value)
    wls_filter.setSigmaColor(sigma_color)

    disp_filt = wls_filter.filter(disp_left, left_gray, None, disp_right)
    disparity = disp_filt.astype(np.float32) / 16.0
    disparity[disparity <= 0] = np.nan

    return disparity


def run_sgbm_wls(left_gray: np.ndarray, right_gray: np.ndarray, matcher=None) -> np.ndarray:
    if matcher is None:
        matcher = create_sgbm_matcher()
    return apply_wls_filter(left_gray, right_gray, matcher)