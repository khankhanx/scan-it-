import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import zipfile
import os

from streamlit_image_coordinates import streamlit_image_coordinates


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Document Scanner",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Smart Document Scanner")
st.caption("CamScanner-style crop + Magic Scan + printer-friendly cleanup")


# =========================================================
# SESSION STATE
# =========================================================

if "pages" not in st.session_state:
    st.session_state.pages = []

if "results" not in st.session_state:
    st.session_state.results = {}

if "crop_points" not in st.session_state:
    st.session_state.crop_points = {}

if "crop_mode" not in st.session_state:
    st.session_state.crop_mode = {}


# =========================================================
# IMAGE CONVERSION
# =========================================================

def pil_to_cv(img):
    img = img.convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# =========================================================
# RESIZE FOR PROCESSING
# =========================================================

def resize_for_detection(image, max_width=1400):

    h, w = image.shape[:2]

    if w <= max_width:
        return image, 1.0

    scale = max_width / float(w)

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


# =========================================================
# ORDER FOUR CORNERS
# =========================================================

def order_points(points):

    pts = np.array(points, dtype=np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


# =========================================================
# AUTOMATIC DOCUMENT DETECTION
# =========================================================

def detect_document(image):

    original = image.copy()

    small, scale = resize_for_detection(image)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Reduce camera noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 40, 150)

    # Strengthen edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = small.shape[:2]
    image_area = h * w

    best = None
    best_score = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.15:
            continue

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.025 * perimeter,
            True
        )

        if len(approx) == 4 and cv2.isContourConvex(approx):

            score = area

            if score > best_score:
                best_score = score
                best = approx.reshape(4, 2)

    if best is None:

        # Fallback: use whole image
        h2, w2 = original.shape[:2]

        return np.array([
            [0, 0],
            [w2 - 1, 0],
            [w2 - 1, h2 - 1],
            [0, h2 - 1]
        ], dtype=np.float32)

    # Convert coordinates back to original resolution
    best = best / scale

    return order_points(best)


# =========================================================
# PERSPECTIVE TRANSFORM
# =========================================================

def four_point_transform(image, points):

    rect = order_points(points)

    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)

    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_height = int(max(height_a, height_b))

    max_width = max(max_width, 300)
    max_height = max(max_height, 300)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        rect.astype(np.float32),
        dst
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped


# =========================================================
# BACKGROUND / SHADOW CLEANUP
# =========================================================

def remove_background(gray):

    # Estimate slowly changing background
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=25,
        sigmaY=25
    )

    gray_float = gray.astype(np.float32) + 1.0
    bg_float = background.astype(np.float32) + 1.0

    # Flatten illumination
    normalized = gray_float / bg_float * 255.0

    normalized = np.clip(
        normalized,
        0,
        255
    ).astype(np.uint8)

    return normalized


# =========================================================
# SHARPEN
# =========================================================

def sharpen(image):

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        1.2
    )

    sharp = cv2.addWeighted(
        image,
        1.45,
        blurred,
        -0.45,
        0
    )

    return np.clip(
        sharp,
        0,
        255
    ).astype(np.uint8)


# =========================================================
# MAGIC SCAN
# =========================================================

def magic_scan(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Remove lighting/shadows
    clean = remove_background(gray)

    # Local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.2,
        tileGridSize=(8, 8)
    )

    clean = clahe.apply(clean)

    # Mild denoise
    clean = cv2.bilateralFilter(
        clean,
        7,
        35,
        35
    )

    # Sharpen text
    clean = sharpen(clean)

    # Slight brightness correction
    clean = cv2.normalize(
        clean,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return cv2.cvtColor(
        clean,
        cv2.COLOR_GRAY2BGR
    )


# =========================================================
# MAGIC CLEAN
# =========================================================

def magic_clean(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clean = remove_background(gray)

    # Stronger local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    clean = clahe.apply(clean)

    clean = cv2.GaussianBlur(
        clean,
        (3, 3),
        0
    )

    clean = sharpen(clean)

    # Adaptive threshold for clean paper
    bw = cv2.adaptiveThreshold(
        clean,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )

    # Small morphology cleanup
    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        kernel
    )

    return cv2.cvtColor(
        bw,
        cv2.COLOR_GRAY2BGR
    )


# =========================================================
# GRAYSCALE HD
# =========================================================

def grayscale_hd(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = remove_background(gray)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    gray = cv2.bilateralFilter(
        gray,
        7,
        30,
        30
    )

    gray = sharpen(gray)

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )


# =========================================================
# PURE BLACK & WHITE
# =========================================================

def pure_bw(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = remove_background(gray)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return cv2.cvtColor(
        bw,
        cv2.COLOR_GRAY2BGR
    )


# =========================================================
# COLOR ENHANCE
# =========================================================

def color_enhance(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge(
        [l, a, b]
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    # Slight sharpening
    enhanced = sharpen(enhanced)

    return enhanced


# =========================================================
# FILTER SELECTOR
# =========================================================

def apply_filter(image, filter_name):

    if filter_name == "Magic Scan":
        return magic_scan(image)

    if filter_name == "Magic Clean":
        return magic_clean(image)

    if filter_name == "Grayscale HD":
        return grayscale_hd(image)

    if filter_name == "Pure B&W":
        return pure_bw(image)

    if filter_name == "Color Enhance":
        return color_enhance(image)

    return image


# =========================================================
# FILE READING
# =========================================================

def read_uploaded_images(uploaded_files):

    images = []

    for uploaded in uploaded_files:

        name = uploaded.name.lower()

        # Normal image
        if name.endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
        ):

            try:

                img = Image.open(
                    uploaded
                ).convert("RGB")

                images.append(
                    (uploaded.name, img)
                )

            except Exception:
                pass

        # ZIP
        elif name.endswith(".zip"):

            try:

                uploaded.seek(0)

                with zipfile.ZipFile(
                    uploaded,
                    "r"
                ) as z:

                    for item in z.infolist():

                        if item.is_dir():
                            continue

                        item_name = item.filename.lower()

                        if item_name.endswith(
                            (".jpg", ".jpeg", ".png",
                             ".webp", ".bmp", ".tiff")
                        ):

                            data = z.read(item)

                            img = Image.open(
                                BytesIO(data)
                            ).convert("RGB")

                            clean_name = os.path.basename(
                                item.filename
                            )

                            images.append(
                                (clean_name, img)
                            )

            except Exception as e:

                st.warning(
                    f"ZIP read error: {uploaded.name} — {e}"
                )

    return images


# =========================================================
# ZIP OUTPUT
# =========================================================

def create_output_zip(results):

    memory_file = BytesIO()

    with zipfile.ZipFile(
        memory_file,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        for filename, image in results:

            buffer = BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=96,
                optimize=True
            )

            z.writestr(
                filename,
                buffer.getvalue()
            )

    memory_file.seek(0)

    return memory_file.getvalue()


# =========================================================
# UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "Images ya ZIP file upload karein",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "tiff",
        "zip"
    ],
    accept_multiple_files=True
)


# =========================================================
# LOAD FILES
# =========================================================

if uploaded_files:

    if st.button(
        "📥 Load Files",
        use_container_width=True
    ):

        loaded = read_uploaded_images(
            uploaded_files
        )

        st.session_state.pages = loaded
        st.session_state.results = {}
        st.session_state.crop_points = {}

        st.success(
            f"{len(loaded)} image(s) loaded."
        )


# =========================================================
# MAIN SCANNER
# =========================================================

if st.session_state.pages:

    st.divider()

    st.subheader(
        f"📑 Pages: {len(st.session_state.pages)}"
    )

    filter_name = st.selectbox(
        "Scan Filter",
        [
            "Magic Scan",
            "Magic Clean",
            "Grayscale HD",
            "Pure B&W",
            "Color Enhance"
        ],
        index=0
    )

    st.info(
        "Magic Scan recommended hai. "
        "Printer ke liye Magic Clean ya Pure B&W use karein."
    )

    # -----------------------------------------------------
    # PROCESS ALL
    # -----------------------------------------------------

    if st.button(
        "✨ Scan All Pages",
        use_container_width=True
    ):

        progress = st.progress(0)

        st.session_state.results = {}

        total = len(
            st.session_state.pages
        )

        for index, (filename, pil_img) in enumerate(
            st.session_state.pages
        ):

            cv_img = pil_to_cv(
                pil_img
            )

            # Automatic corners
            corners = detect_document(
                cv_img
            )

            # Perspective correction
            scanned = four_point_transform(
                cv_img,
                corners
            )

            # Filter
            result = apply_filter(
                scanned,
                filter_name
            )

            st.session_state.results[
                filename
            ] = result

            progress.progress(
                (index + 1) / total
            )

        st.success(
            "All pages scanned successfully."
        )


# =========================================================
# RESULTS
# =========================================================

if st.session_state.results:

    st.divider()

    st.subheader(
        "✂️ Scan Result + Crop"
    )

    st.caption(
        "Crop ke liye image par 4 corners click karein: "
        "Top-Left → Top-Right → Bottom-Right → Bottom-Left"
    )

    final_results = []

    for page_index, (
        filename,
        result
    ) in enumerate(
        st.session_state.results.items()
    ):

        st.markdown(
            f"### Page {page_index + 1}: {filename}"
        )

        display_img = cv_to_pil(
            result
        )

        # Keep display manageable
        display_copy = display_img.copy()

        max_display = 1000

        if display_copy.width > max_display:

            ratio = (
                max_display /
                display_copy.width
            )

            display_copy = display_copy.resize(
                (
                    max_display,
                    int(
                        display_copy.height *
                        ratio
                    )
                )
            )

        # -------------------------------------------------
        # CROP POINT STORAGE
        # -------------------------------------------------

        if filename not in st.session_state.crop_points:

            st.session_state.crop_points[
                filename
            ] = []

        points = st.session_state.crop_points[
            filename
        ]

        # -------------------------------------------------
        # IMAGE CLICK
        # -------------------------------------------------

        clicked = streamlit_image_coordinates(
            display_copy,
            key=f"coords_{page_index}_{filename}"
        )

        if clicked:

            x = clicked["x"]
            y = clicked["y"]

            # Don't add duplicate clicks
            if len(points) < 4:

                points.append(
                    [x, y]
                )

                st.session_state.crop_points[
                    filename
                ] = points

                st.rerun()

        # -------------------------------------------------
        # SHOW SELECTED POINTS
        # -------------------------------------------------

        if len(points) > 0:

            st.write(
                f"Selected corners: {len(points)}/4"
            )

            for i, p in enumerate(points):

                st.write(
                    f"{i + 1}. X={p[0]}  Y={p[1]}"
                )

        col1, col2, col3 = st.columns(3)

        # -------------------------------------------------
        # APPLY CROP
        # -------------------------------------------------

        with col1:

            if st.button(
                "✂️ Apply Crop",
                key=f"crop_{page_index}"
            ):

                if len(points) == 4:

                    # Scale coordinates back
                    scale_x = (
                        result.shape[1] /
                        display_copy.width
                    )

                    scale_y = (
                        result.shape[0] /
                        display_copy.height
                    )

                    real_points = []

                    for x, y in points:

                        real_points.append(
                            [
                                int(x * scale_x),
                                int(y * scale_y)
                            ]
                        )

                    cropped = four_point_transform(
                        result,
                        real_points
                    )

                    st.session_state.results[
                        filename
                    ] = cropped

                    st.session_state.crop_points[
                        filename
                    ] = []

                    st.success(
                        "Crop applied."
                    )

                    st.rerun()

                else:

                    st.warning(
                        "Pehle 4 corners select karein."
                    )

        # -------------------------------------------------
        # RESET CROP
        # -------------------------------------------------

        with col2:

            if st.button(
                "↩️ Reset Crop",
                key=f"reset_{page_index}"
            ):

                st.session_state.crop_points[
                    filename
                ] = []

                st.rerun()

        # -------------------------------------------------
        # DOWNLOAD SINGLE
        # -------------------------------------------------

        with col3:

            buffer = BytesIO()

            cv_to_pil(
                result
            ).save(
                buffer,
                format="JPEG",
                quality=96
            )

            st.download_button(
                "⬇️ Download",
                data=buffer.getvalue(),
                file_name=f"scan_{page_index + 1}.jpg",
                mime="image/jpeg",
                key=f"download_{page_index}"
            )

        st.divider()

        final_results.append(
            (
                f"scan_{page_index + 1}.jpg",
                cv_to_pil(result)
            )
        )

    # =====================================================
    # DOWNLOAD ALL
    # =====================================================

    if final_results:

        zip_data = create_output_zip(
            final_results
        )

        st.download_button(
            "📦 Download ALL Scanned Pages (ZIP)",
            data=zip_data,
            file_name="scanned_documents.zip",
            mime="application/zip",
            use_container_width=True
        )
