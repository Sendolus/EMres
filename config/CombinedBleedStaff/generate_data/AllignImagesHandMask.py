"""
Align recto and verso images, as well as their corresponding masks
"""
import cv2
import numpy as np
import os

# Load and optionally resize images
recto_path = "dataset/EVERYTHING/images/IE9381778_059_24r.jpeg"
rmask_path = "dataset/EVERYTHING/masks/staff_lines/IE9381778_059_24r.jpeg"
verso_path = "dataset/EVERYTHING/images/IE9381778_060_24v.jpeg"
vmask_path = "dataset/EVERYTHING/masks/staff_lines/IE9381778_060_24v.jpeg"

recto_orig = cv2.imread(recto_path)
rmask_orig = cv2.imread(rmask_path)
verso_orig = cv2.flip(cv2.imread(verso_path), 1)  # Flip verso horizontally
vmask_orig = cv2.flip(cv2.imread(vmask_path), 1)
H_orig, W_orig = recto_orig.shape[0], recto_orig.shape[1]

recto = cv2.resize(recto_orig, (720, 1080))
rmask = cv2.resize(rmask_orig, (720, 1080))
verso = cv2.resize(verso_orig, (720, 1080))
vmask = cv2.resize(vmask_orig, (720, 1080))

points_recto = []
points_verso = []
new_points_verso = []
click_stage = 0  # 0 = recto, 1 = verso


def click_points(event, x, y, flags, param):
    global click_stage
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_stage % 2 == 0:
            points_recto.append([x, y])
            print(f"Recto point {len(points_recto)}: {x}, {y}")
        else:
            points_verso.append([x, y])
            print(f"Verso point {len(points_verso)}: {x}, {y}")
        click_stage += 1


cv2.namedWindow("Image Selection")
cv2.setMouseCallback("Image Selection", click_points)
print("Click a point on RECTO, then its match on VERSO, then next RECTO, etc... Press ENTER when done.")

while True:
    combined = np.hstack((recto.copy(), verso.copy()))
    offset = recto.shape[1]

    for i, pt in enumerate(points_recto):
        cv2.circle(combined, tuple(pt), 5, (0, 0, 255), -1)
        cv2.putText(combined, f"R{i+1}", tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for i, pt in enumerate(points_verso):
        verso_pt = (pt[0], pt[1])
        cv2.circle(combined, tuple(verso_pt), 5, (0, 255, 0), -1)
        cv2.putText(combined, f"V{i+1}", tuple(verso_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Image alignment", combined)
    key = cv2.waitKey(1)
    if key == 13:  # ENTER key
        break
    elif key == ord("z") and click_stage > 0:  # Undo last point
        click_stage -= 1
        if click_stage % 2 == 0:
            points_recto.pop()
        else:
            points_verso.pop()

cv2.destroyAllWindows()

for point in points_verso:
    test = [point[0]-offset, point[1]]
    new_points_verso.append(test)
points_verso = new_points_verso

# Homography and warp
if 4 <= len(points_recto) == len(points_verso):
    pts1 = np.array(points_recto, dtype=np.float32)
    pts2 = np.array(points_verso, dtype=np.float32)
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    aligned = cv2.warpPerspective(verso, H, (recto.shape[1], recto.shape[0]))
    blended = cv2.addWeighted(aligned, 0.5, recto, 0.5, 0)
    cv2.imshow("Aligned & Blended", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save resized versions of the resized and aligned images
    aligned_mask = cv2.warpPerspective(vmask, H, (recto.shape[1], recto.shape[0]))
    cv2.imwrite("vmask.jpg", aligned_mask)
    cv2.imwrite("rmask.jpg", rmask)
    cv2.imwrite("recto.jpg", recto)
    cv2.imwrite("verso.jpg", aligned)

    # Save the warped original image
    scale_x = W_orig / 720
    scale_y = H_orig / 1080

    # Scale matrix for pre-warp
    S = np.array([[scale_x, 0, 0],
                  [0, scale_y, 0],
                  [0, 0, 1]])
    H_original = S @ H @ np.linalg.inv(S)
    H_inv = np.linalg.inv(H_original)

    aligned_full_res = cv2.warpPerspective(verso_orig, H_original, (recto_orig.shape[1], recto_orig.shape[0]))
else:
    print("Not enough matching points (need at least 4), or point count mismatch.")