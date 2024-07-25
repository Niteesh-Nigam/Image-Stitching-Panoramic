import numpy as np
import cv2

class PanoramaCreator:
    def stitch(self, images, lowe_ratio=0.75, ransac_thresh=4.0, display_matches=False):
        # Extract features and keypoints
        (imgB, imgA) = images
        (kpA, featA) = self._extract_features(imgA)
        (kpB, featB) = self._extract_features(imgB)

        # Match keypoints and compute homography
        match_info = self._match_keypoints(kpA, kpB, featA, featB, lowe_ratio, ransac_thresh)
        if match_info is None:
            return None

        # Warp perspective and combine images
        (matches, H, status) = match_info
        result = self._warp_images(imgA, imgB, H)
        result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB

        if display_matches:
            vis = self._draw_matches(imgA, imgB, kpA, kpB, matches, status)
            return result, vis

        return result

    def _warp_images(self, imgA, imgB, H):
        width = imgA.shape[1] + imgB.shape[1]
        result = cv2.warpPerspective(imgA, H, (width, imgA.shape[0]))
        return result

    def _extract_features(self, image):
        sift = cv2.SIFT_create()
        keypoints, features = sift.detectAndCompute(image, None)
        keypoints = np.float32([kp.pt for kp in keypoints])
        return keypoints, features

    def _match_keypoints(self, kpA, kpB, featA, featB, lowe_ratio, ransac_thresh):
        all_matches = self._compute_matches(featA, featB)
        valid_matches = self._filter_matches(all_matches, lowe_ratio)

        if len(valid_matches) > 4:
            ptsA = np.float32([kpA[i] for (_, i) in valid_matches])
            ptsB = np.float32([kpB[i] for (i, _) in valid_matches])
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransac_thresh)
            return valid_matches, H, status
        return None

    def _compute_matches(self, featA, featB):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        matches = matcher.knnMatch(featA, featB, 2)
        return matches

    def _filter_matches(self, matches, lowe_ratio):
        valid_matches = []
        for m, n in matches:
            if m.distance < n.distance * lowe_ratio:
                valid_matches.append((m.trainIdx, m.queryIdx))
        return valid_matches

    def _draw_matches(self, imgA, imgB, kpA, kpB, matches, status):
        (hA, wA) = imgA.shape[:2]
        vis = np.zeros((max(hA, imgB.shape[0]), wA + imgB.shape[1], 3), dtype="uint8")
        vis[0:hA, 0:wA] = imgA
        vis[0:imgB.shape[0], wA:] = imgB

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpA[queryIdx][0]), int(kpA[queryIdx][1]))
                ptB = (int(kpB[trainIdx][0]) + wA, int(kpB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        return vis
