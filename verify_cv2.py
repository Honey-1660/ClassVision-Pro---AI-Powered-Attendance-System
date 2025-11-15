import cv2
import sys

print('Python executable:', sys.executable)
print('cv2 import attempt...')
try:
    print('cv2 version:', cv2.__version__)
    has_face = hasattr(cv2, 'face')
    print('has face module:', has_face)
    has_lbph = False
    if has_face:
        try:
            has_lbph = hasattr(cv2.face, 'LBPHFaceRecognizer_create')
        except Exception as e:
            print('Error checking LBPH:', e)
            has_lbph = False
    print('has LBPHFaceRecognizer_create:', has_lbph)
except Exception as exc:
    print('Failed to import or inspect cv2:', exc)
    raise