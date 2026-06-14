import cv2
import matplotlib.pyplot as plt
image = cv2.imread(
    "data/sample_docs/handwritten_note.jpeg"
)

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

gray = cv2.GaussianBlur(
    gray,
    (5,5),
    0
)

thresh = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    21,
    10
)

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

image_copy = image.copy()
count = 0

for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    print(
        f"x={x}, y={y}, w={w}, h={h}"
    )

    aspect_ratio = w / h

    if (
        w > 20
        and h > 40
        and aspect_ratio < 3
    ):

        count += 1

        crop = thresh[y:y+h, x:x+w]

        cv2.imwrite(
            f"outputs/char_{count}.png",
            crop
        )

        cv2.rectangle(
            image_copy,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
print("Detected Characters:", count)
print("Original:", image.shape)
print("Gray:", gray.shape)

plt.figure(figsize=(8,8))

plt.imshow(
    cv2.cvtColor(
        image_copy,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Detected Regions")
plt.axis("off")

plt.show()