"""
Generate a professional PDF from the COMPLETE_GUIDE.md for the QArm Fruit Sorting project.
Uses ReportLab for PDF generation.
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, gray
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, KeepTogether, HRFlowable
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.fonts import addMapping

# --- Colors ---
BLUE_DARK = HexColor('#1a365d')
BLUE_MED = HexColor('#2b6cb0')
BLUE_LIGHT = HexColor('#ebf4ff')
GREEN = HexColor('#276749')
GREEN_LIGHT = HexColor('#f0fff4')
RED = HexColor('#c53030')
RED_LIGHT = HexColor('#fff5f5')
ORANGE = HexColor('#c05621')
ORANGE_LIGHT = HexColor('#fffaf0')
GRAY_BG = HexColor('#f7fafc')
GRAY_BORDER = HexColor('#cbd5e0')
GRAY_TEXT = HexColor('#4a5568')

# --- Output path ---
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "COMPLETE_GUIDE.pdf")

def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'CoverTitle', parent=styles['Title'],
        fontSize=28, leading=34, textColor=BLUE_DARK,
        spaceAfter=8, alignment=TA_CENTER, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        'CoverSubtitle', parent=styles['Normal'],
        fontSize=14, leading=18, textColor=BLUE_MED,
        spaceAfter=6, alignment=TA_CENTER, fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        'CoverInfo', parent=styles['Normal'],
        fontSize=11, leading=15, textColor=GRAY_TEXT,
        spaceAfter=4, alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        'SectionHeader', parent=styles['Heading1'],
        fontSize=18, leading=22, textColor=BLUE_DARK,
        spaceBefore=20, spaceAfter=10, fontName='Helvetica-Bold',
        borderWidth=1, borderColor=BLUE_MED, borderPadding=5,
    ))
    styles.add(ParagraphStyle(
        'SubHeader', parent=styles['Heading2'],
        fontSize=14, leading=17, textColor=BLUE_MED,
        spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        'SubSubHeader', parent=styles['Heading3'],
        fontSize=12, leading=15, textColor=HexColor('#2d3748'),
        spaceBefore=10, spaceAfter=4, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        'BodyText2', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=HexColor('#2d3748'),
        spaceAfter=6, alignment=TA_JUSTIFY, fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        'CodeBlock', parent=styles['Normal'],
        fontSize=8.5, leading=11, textColor=HexColor('#1a202c'),
        fontName='Courier', backColor=GRAY_BG,
        borderWidth=0.5, borderColor=GRAY_BORDER, borderPadding=6,
        spaceBefore=4, spaceAfter=6, leftIndent=10
    ))
    styles.add(ParagraphStyle(
        'BulletItem', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=HexColor('#2d3748'),
        leftIndent=20, bulletIndent=10, spaceAfter=3
    ))
    styles.add(ParagraphStyle(
        'CheckItem', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=HexColor('#2d3748'),
        leftIndent=20, bulletIndent=10, spaceAfter=3
    ))
    styles.add(ParagraphStyle(
        'NoteBox', parent=styles['Normal'],
        fontSize=9.5, leading=13, textColor=ORANGE,
        backColor=ORANGE_LIGHT, borderWidth=0.5, borderColor=ORANGE,
        borderPadding=8, spaceBefore=6, spaceAfter=8, leftIndent=5
    ))
    styles.add(ParagraphStyle(
        'ImportantBox', parent=styles['Normal'],
        fontSize=9.5, leading=13, textColor=RED,
        backColor=RED_LIGHT, borderWidth=0.5, borderColor=RED,
        borderPadding=8, spaceBefore=6, spaceAfter=8, leftIndent=5
    ))
    styles.add(ParagraphStyle(
        'TableHeader', parent=styles['Normal'],
        fontSize=9, leading=12, textColor=white,
        fontName='Helvetica-Bold', alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        'TableCell', parent=styles['Normal'],
        fontSize=9, leading=12, textColor=HexColor('#2d3748'),
        fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        'Footer', parent=styles['Normal'],
        fontSize=8, textColor=GRAY_TEXT, alignment=TA_CENTER
    ))
    return styles


def make_table(headers, rows, col_widths=None):
    """Create a styled table."""
    s = build_styles()
    header_cells = [Paragraph(f"<b>{h}</b>", s['TableHeader']) for h in headers]
    data = [header_cells]
    for row in rows:
        data.append([Paragraph(str(c), s['TableCell']) for c in row])

    if col_widths is None:
        col_widths = [None] * len(headers)

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE_DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, GRAY_BG]),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2d3748')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, GRAY_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


def add_footer(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont('Helvetica', 8)
    canvas_obj.setFillColor(GRAY_TEXT)
    canvas_obj.drawCentredString(
        A4[0] / 2, 15 * mm,
        f"QArm Fruit Sorting - Complete Guide | Page {doc.page}"
    )
    canvas_obj.restoreState()


def build_pdf():
    s = build_styles()
    doc = SimpleDocTemplate(
        OUTPUT_PATH, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=25*mm,
        title="QArm Fruit Sorting - Complete Guide",
        author="Applied Robotics Team"
    )
    W = A4[0] - 40*mm  # usable width

    story = []

    # ===== COVER PAGE =====
    story.append(Spacer(1, 60*mm))
    story.append(Paragraph("Complete Step-by-Step Guide", s['CoverTitle']))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("QArm Fruit Sorting Final Project", s['CoverSubtitle']))
    story.append(Spacer(1, 10*mm))
    story.append(HRFlowable(width="60%", thickness=2, color=BLUE_MED, spaceBefore=5, spaceAfter=15))
    story.append(Paragraph("Applied Robotics (04 39984)", s['CoverInfo']))
    story.append(Paragraph("University of Birmingham", s['CoverInfo']))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("Team: Piero Flores, Zihen Huang, Ran Zhang, Yichang Chao", s['CoverInfo']))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("<b>Deadline:</b> Friday 1 May 2026, 2:00 PM", s['CoverInfo']))
    story.append(PageBreak())

    # ===== TABLE OF CONTENTS =====
    story.append(Paragraph("TABLE OF CONTENTS", s['SectionHeader']))
    story.append(Spacer(1, 5*mm))
    toc_items = [
        ("1.", "Project Overview"),
        ("2.", "What Has Been Completed (Software)"),
        ("3.", "What Must Be Done Manually (In Lab)"),
        ("4.", "Step-by-Step Lab Procedure"),
        ("5.", "Building the Simulink Model (Hardware)"),
        ("6.", "Camera Calibration Procedure"),
        ("7.", "Running Autonomous Mode"),
        ("8.", "Running Remote Control Mode"),
        ("9.", "Testing and Validation"),
        ("10.", "Report Writing Guide"),
        ("11.", "Presentation Guide"),
        ("12.", "Troubleshooting"),
    ]
    for num, title in toc_items:
        story.append(Paragraph(f"<b>{num}</b> {title}", s['BodyText2']))
    story.append(PageBreak())

    # ===== SECTION 1: PROJECT OVERVIEW =====
    story.append(Paragraph("1. PROJECT OVERVIEW", s['SectionHeader']))
    story.append(Paragraph("What the Project Requires:", s['SubHeader']))
    for item in [
        "Sort 14 fruits (6 strawberries, 3 bananas, 5 tomatoes) into 3 baskets",
        "<b>Autonomous mode:</b> Robot detects, picks, and sorts fruits automatically using RGBD camera",
        "<b>Remote control mode:</b> Operator controls the robot via keyboard interface",
        "Uses the Quanser QArm 4-DOF robot with Intel RealSense D415 RGBD camera",
    ]:
        story.append(Paragraph(f"\u2022 {item}", s['BulletItem']))

    story.append(Paragraph("Deliverables:", s['SubHeader']))
    story.append(make_table(
        ["#", "Deliverable", "Due Date", "Weight"],
        [
            ["1", "PowerPoint Slides (5 max per assignment)", "With presentation", "5% each"],
            ["2", "Final Presentation (12 min + 3 min Q&A)", "TBC (Week 6)", "15%"],
            ["3", "Final Report (22 pages)", "1 May 2026", "65%"],
            ["4", "Peer Assessment", "1 May 2026", "5%"],
        ],
        col_widths=[20*mm, 70*mm, 40*mm, 30*mm]
    ))
    story.append(PageBreak())

    # ===== SECTION 2: COMPLETED SOFTWARE =====
    story.append(Paragraph("2. WHAT HAS BEEN COMPLETED (Software)", s['SectionHeader']))
    story.append(Paragraph(
        "All MATLAB/Simulink code has been written and validated in offline simulation. "
        "The following files are ready to be integrated into the hardware Simulink model.",
        s['BodyText2']
    ))

    story.append(Paragraph("Core MATLAB Functions", s['SubHeader']))
    story.append(make_table(
        ["File", "Purpose", "Status"],
        [
            ["qarm_FK.m", "Forward kinematics (compact DH convention)", "VALIDATED, exact"],
            ["qarm_IK.m", "Inverse kinematics (analytical + Newton-Raphson)", "VALIDATED, &lt;0.001mm"],
            ["cubicTrajectory.m", "Single-segment cubic spline trajectory", "VALIDATED"],
            ["multiSegmentTrajectory.m", "Multi-waypoint trajectory planner", "VALIDATED"],
            ["fruitDetector.m", "HSV color-based fruit detection/classification", "VALIDATED"],
            ["pixelToWorld.m", "Camera pixel+depth to world coordinates", "Needs calibration"],
            ["fruitSortingController.m", "Full 14-state autonomous controller", "VALIDATED (67s for 6 fruits)"],
        ],
        col_widths=[55*mm, 60*mm, 45*mm]
    ))

    story.append(Paragraph("Scripts", s['SubHeader']))
    story.append(make_table(
        ["File", "Purpose"],
        [
            ["create_FruitSorting_Model.m", "Generates the base Simulink model skeleton"],
            ["calibrate_camera.m", "Camera-to-robot calibration tool (SVD-based)"],
            ["validate_fruit_sorting.m", "Full offline simulation + generates 5 report figures"],
            ["test_fruit_detection.m", "Detection unit test with synthetic images"],
        ],
        col_widths=[60*mm, 100*mm]
    ))

    story.append(Paragraph("Simulation Results Summary", s['SubHeader']))
    for item in [
        "<b>6 fruits sorted in 66.9 seconds</b> (offline simulation)",
        "FK/IK round-trip error: <b>&lt; 0.001 mm</b> (sub-micron accuracy)",
        "Trajectory velocities: within safe limits (&lt; 0.3 m/s)",
        "Zero-velocity boundary conditions on all trajectory segments",
        "5 publication-ready figures generated for the report",
    ]:
        story.append(Paragraph(f"\u2022 {item}", s['BulletItem']))
    story.append(PageBreak())

    # ===== SECTION 3: MANUAL TASKS =====
    story.append(Paragraph("3. WHAT MUST BE DONE MANUALLY (In Lab)", s['SectionHeader']))
    story.append(Paragraph(
        "<b>CRITICAL:</b> These tasks REQUIRE physical access to the QArm robot and cannot be done remotely.",
        s['ImportantBox']
    ))
    checklist = [
        ("3.1", "Set up the physical workspace (fruits and baskets)"),
        ("3.2", "Build the Simulink model for QUARC hardware deployment"),
        ("3.3", "Calibrate the camera (T_cam_to_base transform)"),
        ("3.4", "Tune HSV color thresholds for actual lab lighting"),
        ("3.5", "Test autonomous mode on real hardware"),
        ("3.6", "Test remote control mode on real hardware"),
        ("3.7", "Record videos/screenshots for report and presentation"),
        ("3.8", "Design soft gripper (report section: analytical + conceptual)"),
        ("3.9", "Write the 22-page report"),
        ("3.10", "Prepare final presentation slides"),
    ]
    for num, desc in checklist:
        story.append(Paragraph(f"\u2610 <b>{num}</b> - {desc}", s['CheckItem']))
    story.append(PageBreak())

    # ===== SECTION 4: LAB PROCEDURE =====
    story.append(Paragraph("4. STEP-BY-STEP LAB PROCEDURE", s['SectionHeader']))

    story.append(Paragraph("4.1 - Workspace Setup (15 min)", s['SubHeader']))
    steps_41 = [
        "Power on the QArm and connect USB to the lab PC",
        "Open MATLAB R2025a",
        "Verify QUARC is installed: type <font face='Courier' size='9'>quarc_setup</font> in MATLAB command window",
        "Copy the <font face='Courier' size='9'>FinalProject_FruitSorting/</font> folder to the lab PC",
        "Add matlab_code/ to MATLAB path: <font face='Courier' size='9'>addpath('FinalProject_FruitSorting/matlab_code');</font>",
    ]
    for i, step in enumerate(steps_41, 1):
        story.append(Paragraph(f"<b>{i}.</b> {step}", s['BulletItem']))

    story.append(Paragraph("<b>6. Arrange the physical workspace:</b>", s['BulletItem']))
    story.append(Paragraph(
        "Place 3 baskets around the QArm as follows:\n"
        "<font face='Courier' size='8'>"
        "&nbsp;&nbsp;&nbsp;&nbsp;Strawberry basket: (0.30, -0.20, 0.05) - Right-front<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;Banana basket: (-0.30, -0.20, 0.05) - Left-front<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;Tomato basket: (0.00, -0.35, 0.05) - Center-front<br/>"
        "</font>"
        "Place fruits randomly in region Y=[0.15, 0.40], X=[-0.25, 0.25], at table height (~2cm).",
        s['CodeBlock']
    ))

    story.append(Paragraph("4.2 - Initial Hardware Test (10 min)", s['SubHeader']))
    steps_42 = [
        "Open <font face='Courier' size='9'>BasicIO_position_mode.slx</font> from Lab 0 courseware",
        "Build and run with QUARC: <b>Monitor &amp; Tune</b>",
        "Verify the robot moves to home position [0.45, 0, 0.49]",
        "Test gripper open/close",
        "Stop the model",
    ]
    for i, step in enumerate(steps_42, 1):
        story.append(Paragraph(f"<b>{i}.</b> {step}", s['BulletItem']))
    story.append(PageBreak())

    # ===== SECTION 5: SIMULINK MODEL =====
    story.append(Paragraph("5. BUILDING THE SIMULINK MODEL (Hardware)", s['SectionHeader']))
    story.append(Paragraph(
        "<b>RECOMMENDED:</b> Modify the existing Lab 5 PickAndPlace.slx model. "
        "This is the fastest approach since the QUARC hardware interface is already configured.",
        s['NoteBox']
    ))

    steps_5 = [
        ("<b>Copy</b> <font face='Courier' size='9'>PickAndPlace.slx</font> from "
         "Lab 5 Hardware courseware to <font face='Courier' size='9'>FinalProject_FruitSorting/simulink_models/</font>"),
        "<b>Open</b> the copied model in Simulink",
        ("<b>Replace</b> the <font face='Courier' size='9'>qarmForwardKinematics</font> MATLAB Function block "
         "content with the body of <font face='Courier' size='9'>qarm_FK.m</font>"),
        ("<b>Replace</b> the <font face='Courier' size='9'>qarmInverseKinematics</font> MATLAB Function block "
         "content with the body of <font face='Courier' size='9'>qarm_IK.m</font>"),
        ("<b>Add Vision System:</b> Copy camera interface from Lab 9 "
         "<font face='Courier' size='9'>ImageAcquisitionAndColorSpaces.slx</font>. "
         "Add MATLAB Function blocks for <font face='Courier' size='9'>fruitDetector.m</font> and "
         "<font face='Courier' size='9'>pixelToWorld.m</font>"),
        ("<b>Add Controller:</b> Create a MATLAB Function block with "
         "<font face='Courier' size='9'>fruitSortingController.m</font>. "
         "Connect: joint_pos, ee_pos, fruit_positions, fruit_types, mode, operator_cmd, Clock"),
        ("<b>Add Mode Switch:</b> Manual Switch block between Autonomous and Remote modes"),
        ("<b>Configuration:</b> Solver=Fixed-step, ode4, Step=0.002s (500Hz), StopTime=300s"),
    ]
    for i, step in enumerate(steps_5, 1):
        story.append(Paragraph(f"<b>Step {i}:</b> {step}", s['BulletItem']))

    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("Signal Flow Diagram:", s['SubSubHeader']))
    story.append(Paragraph(
        "<font face='Courier' size='8'>"
        "[Camera] --&gt; [fruitDetector] --&gt; [pixelToWorld] --&gt; fruit_positions<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br/>"
        "[Clock] --------&gt; [fruitSortingController] &lt;--- mode switch<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "joint_cmd&nbsp;&nbsp;gripper&nbsp;&nbsp;state<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "[QArm Plant + Gripper] --&gt; joint_pos --&gt; [qarm_FK] --&gt; ee_pos<br/>"
        "</font>",
        s['CodeBlock']
    ))
    story.append(PageBreak())

    # ===== SECTION 6: CAMERA CALIBRATION =====
    story.append(Paragraph("6. CAMERA CALIBRATION PROCEDURE", s['SectionHeader']))
    story.append(Paragraph(
        "<b>This is CRITICAL and must be done in the lab!</b> The calibration determines "
        "how pixel coordinates from the camera map to real-world positions in the QArm base frame.",
        s['ImportantBox']
    ))

    story.append(Paragraph("Procedure:", s['SubHeader']))
    calib_steps = [
        "Open <font face='Courier' size='9'>ImageAcquisitionAndColorSpaces.slx</font> from Lab 9",
        "Build and run with QUARC -- verify RGB and depth images appear in scopes",
        "Place a bright colored ball at a KNOWN position (measure with ruler from QArm base center)",
        "Record: pixel_col, pixel_row, depth (from depth image), world_x, world_y, world_z",
        "Repeat for at least <b>6 positions</b> spread across the workspace (front-left, front-right, center, etc.)",
        "Edit <font face='Courier' size='9'>calibrate_camera.m</font>: replace the example calib_data matrix with your measurements",
        "Run the script -- check that reprojection error is <b>&lt; 10mm</b>",
        "Save the <font face='Courier' size='9'>camera_calibration.mat</font> file",
    ]
    for i, step in enumerate(calib_steps, 1):
        story.append(Paragraph(f"<b>{i}.</b> {step}", s['BulletItem']))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "<b>Usage after calibration:</b><br/>"
        "<font face='Courier' size='9'>"
        "load('camera_calibration.mat');<br/>"
        "p_world = pixelToWorld(centroid_px, depth_image, cam_intrinsics, T_cam_to_base);"
        "</font>",
        s['CodeBlock']
    ))
    story.append(PageBreak())

    # ===== SECTION 7: AUTONOMOUS MODE =====
    story.append(Paragraph("7. RUNNING AUTONOMOUS MODE", s['SectionHeader']))

    story.append(Paragraph("State Machine Flow:", s['SubHeader']))
    story.append(Paragraph(
        "<font face='Courier' size='8'>"
        "INIT --&gt; GO_HOME --&gt; SELECT_FRUIT --&gt; APPROACH_FRUIT --&gt; DESCEND_TO_PICK<br/>"
        "&nbsp;&nbsp;^&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br/>"
        "&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;DONE &lt;-- GO_HOME &lt;-- ASCEND_FROM_PLACE &lt;--"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br/>"
        "&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;^"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CLOSE_GRIPPER<br/>"
        "&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;OPEN_GRIPPER"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br/>"
        "&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;^"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ASCEND_FROM_PICK<br/>"
        "&nbsp;&nbsp;+--- (repeat for each fruit) ---"
        "DESCEND_PLACE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MOVE_TO_BASKET<br/>"
        "</font>",
        s['CodeBlock']
    ))

    story.append(Paragraph("Running:", s['SubHeader']))
    auto_steps = [
        "Set the Manual Switch to <b>Autonomous</b>",
        "Place fruits in the workspace",
        "Build and deploy with QUARC (Monitor &amp; Tune)",
        "The robot will automatically detect, pick, and sort all fruits",
    ]
    for i, step in enumerate(auto_steps, 1):
        story.append(Paragraph(f"<b>{i}.</b> {step}", s['BulletItem']))

    story.append(Paragraph("Key Parameters to Tune:", s['SubHeader']))
    story.append(make_table(
        ["Parameter", "Default", "Description"],
        [
            ["safe_z", "0.20 m", "Safe height for transit moves"],
            ["pick_z", "0.02 m", "Height at which to pick fruit"],
            ["place_z", "0.10 m", "Height above basket to release"],
            ["T_transit", "2.0 s", "Duration for large moves"],
            ["T_approach", "1.0 s", "Duration for approach/retreat"],
            ["T_pick", "0.8 s", "Duration for final descent"],
            ["T_dwell", "0.5 s", "Gripper actuation wait time"],
        ],
        col_widths=[40*mm, 25*mm, 95*mm]
    ))
    story.append(Paragraph(
        "<b>IMPORTANT:</b> Adjust <font face='Courier' size='9'>basket_positions</font> in "
        "<font face='Courier' size='9'>fruitSortingController.m</font> to match your actual basket placement!",
        s['ImportantBox']
    ))
    story.append(PageBreak())

    # ===== SECTION 8: REMOTE CONTROL =====
    story.append(Paragraph("8. RUNNING REMOTE CONTROL MODE", s['SectionHeader']))
    story.append(Paragraph(
        "The remote mode reuses the keyboard teleoperation architecture from Lab 3.",
        s['BodyText2']
    ))
    story.append(Paragraph("Controls:", s['SubHeader']))
    story.append(make_table(
        ["Key", "Action"],
        [
            ["Up Arrow", "Move -X (away from operator)"],
            ["Down Arrow", "Move +X (toward operator)"],
            ["Left Arrow", "Move -Y (left)"],
            ["Right Arrow", "Move +Y (right)"],
            ["Q", "Move +Z (up)"],
            ["A", "Move -Z (down)"],
            ["Space", "Toggle gripper (open/close)"],
        ],
        col_widths=[40*mm, 120*mm]
    ))
    story.append(Paragraph(
        "Speed gain: 0.05 m/s per key press. The controller converts velocity commands "
        "to position increments at 500 Hz.",
        s['BodyText2']
    ))
    story.append(PageBreak())

    # ===== SECTION 9: TESTING =====
    story.append(Paragraph("9. TESTING AND VALIDATION", s['SectionHeader']))

    story.append(Paragraph("Pre-deployment Checklist:", s['SubHeader']))
    checks = [
        "Camera calibration completed and saved",
        "HSV thresholds tuned for actual lighting (edit fruitDetector.m)",
        "Basket positions measured and updated in fruitSortingController.m",
        "Fruit pick height verified (fruits sit at correct z height)",
        "Gripper force adequate for all fruit types",
        "Emergency stop (E-stop) location known",
    ]
    for c in checks:
        story.append(Paragraph(f"\u2610 {c}", s['CheckItem']))

    story.append(Paragraph("Test Sequence:", s['SubHeader']))
    tests = [
        ("<b>Single fruit test:</b> Place ONE tomato. Run autonomous mode. "
         "Verify: detection, approach, pick, transit, place. Record video."),
        ("<b>Multi-fruit test:</b> Place one of each type. Run autonomous mode. "
         "Verify correct basket assignment."),
        ("<b>Full test:</b> Place all 14 fruits. Run autonomous mode. "
         "Time the operation. Record video."),
        ("<b>Remote control test:</b> Switch to remote mode. "
         "Manually sort 3 fruits. Record video."),
    ]
    for i, t in enumerate(tests, 1):
        story.append(Paragraph(f"<b>{i}.</b> {t}", s['BulletItem']))

    story.append(Paragraph("Expected Performance:", s['SubHeader']))
    story.append(make_table(
        ["Metric", "Expected Value"],
        [
            ["Single fruit cycle", "~10-12 seconds"],
            ["Full 14-fruit sort", "~2-3 minutes"],
            ["Position accuracy", "&lt; 5mm (with calibrated camera)"],
            ["Detection accuracy", "&gt; 90% (tuned HSV thresholds)"],
        ],
        col_widths=[60*mm, 100*mm]
    ))
    story.append(PageBreak())

    # ===== SECTION 10: REPORT =====
    story.append(Paragraph("10. REPORT WRITING GUIDE", s['SectionHeader']))
    story.append(Paragraph("Structure (22 pages total):", s['SubHeader']))

    story.append(Paragraph("<b>Part 1: Design and Simulation (12 pages)</b>", s['SubSubHeader']))
    report_p1 = [
        ("Title Page", "1 page", "Project title, team members, module code, date"),
        ("Introduction", "1.5 pages", "Objectives, QArm specs, problem statement"),
        ("Kinematics", "2 pages", "DH convention, FK/IK derivation, validation results (0.000mm)"),
        ("Trajectory Generation", "1.5 pages", "Cubic spline formulation, velocity/acceleration analysis"),
        ("Vision System", "2 pages", "RealSense D415 specs, HSV segmentation, calibration"),
        ("Control Architecture", "2 pages", "State machine diagram, auto/remote modes, Simulink block diagram"),
        ("Soft Gripper Design", "2 pages", "Conceptual design, grip force analysis, material selection"),
    ]
    story.append(make_table(
        ["Section", "Length", "Content"],
        report_p1,
        col_widths=[40*mm, 25*mm, 95*mm]
    ))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("<b>Part 2: Validation (7 pages)</b>", s['SubSubHeader']))
    report_p2 = [
        ("Simulation Results", "2 pages", "Use generated figures: 3D trajectory, joint angles, gripper states"),
        ("Hardware Results", "2.5 pages", "Photos from QArm tests, sim vs hardware comparison, success rate"),
        ("Discussion", "1.5 pages", "Discrepancies, limitations, challenges and solutions"),
        ("Conclusion", "1 page", "Summary, recommendations, future work extensions"),
    ]
    story.append(make_table(
        ["Section", "Length", "Content"],
        report_p2,
        col_widths=[40*mm, 25*mm, 95*mm]
    ))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "<b>+ 1 page</b> for feedback reflections. Total = 22 pages.",
        s['BodyText2']
    ))

    story.append(Paragraph("Key Figures Available:", s['SubHeader']))
    figs = [
        ("trajectory_3D.png", "Complete 3D sorting path visualization"),
        ("ee_position_time.png", "End-effector X/Y/Z position over time"),
        ("joint_angles_time.png", "All 4 joint angles during sorting operation"),
        ("gripper_state.png", "Gripper open/close state timeline"),
        ("top_view.png", "Top-down workspace layout with fruit/basket positions"),
    ]
    story.append(make_table(
        ["Figure File", "Description"],
        figs,
        col_widths=[55*mm, 105*mm]
    ))
    story.append(PageBreak())

    # ===== SECTION 11: PRESENTATION =====
    story.append(Paragraph("11. PRESENTATION GUIDE", s['SectionHeader']))
    story.append(Paragraph(
        "Final Presentation: <b>12 minutes + 3 min Q&amp;A</b> (covers all 3 assignments combined)",
        s['BodyText2']
    ))
    story.append(Paragraph("Slide Structure for Final Project:", s['SubHeader']))
    slides = [
        "System Architecture (block diagram, state machine)",
        "Vision + Detection Results (camera images, bounding boxes)",
        "Autonomous Demo Results (trajectory plot, video screenshot)",
        "Remote Control Demo",
        "Challenges, Solutions, and Conclusions",
    ]
    for i, sl in enumerate(slides, 1):
        story.append(Paragraph(f"<b>Slide {i}:</b> {sl}", s['BulletItem']))

    story.append(Paragraph("Videos to Prepare:", s['SubHeader']))
    for v in [
        "Autonomous mode sorting 3+ fruits (~30 second clip)",
        "Remote control mode sorting 1 fruit (~15 second clip)",
        "Detection visualization showing bounding boxes on camera feed",
    ]:
        story.append(Paragraph(f"\u2022 {v}", s['BulletItem']))
    story.append(PageBreak())

    # ===== SECTION 12: TROUBLESHOOTING =====
    story.append(Paragraph("12. TROUBLESHOOTING", s['SectionHeader']))

    issues = [
        ("IK solution not reaching target",
         "The Newton-Raphson refinement ensures &lt;0.001mm error. "
         "If position is unreachable, check it's within 0.75m reach radius. "
         "Verify position isn't in a singularity (directly above base)."),
        ("Fruit not detected",
         "HSV thresholds need tuning for your specific lighting. "
         "Edit fruitDetector.m and adjust H, S, V ranges. "
         "Use Lab 9 model to view camera feed and test interactively."),
        ("Camera calibration error &gt; 10mm",
         "Use more calibration points (minimum 6, ideally 10+). "
         "Spread points across the full workspace. "
         "Ensure depth readings are accurate."),
        ("Robot moves too fast or jerky",
         "Increase T_transit and T_approach in the controller. "
         "Cubic spline ensures zero start/end velocity. "
         "Verify simulation step is 0.002s (500 Hz)."),
        ("Gripper doesn't close properly",
         "Adjust threshold distance in controller (default 3cm). "
         "Verify gripper PWM signal is connected. "
         "Check that gripper_cmd output reaches the plant."),
        ("QUARC build fails",
         "Ensure QUARC license is active. "
         "Check solver is Fixed-step, ode4, 0.002s. "
         "Verify all QUARC blocks are connected. "
         "Check MATLAB path includes matlab_code/."),
    ]
    for title, fix in issues:
        story.append(Paragraph(f"<b>\"{title}\"</b>", s['SubSubHeader']))
        story.append(Paragraph(fix, s['BodyText2']))
    story.append(PageBreak())

    # ===== FILE REFERENCE + TIMELINE =====
    story.append(Paragraph("QUICK REFERENCE: FILE LOCATIONS", s['SectionHeader']))
    story.append(Paragraph(
        "<font face='Courier' size='8'>"
        "FinalProject_FruitSorting/<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;matlab_code/<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;qarm_FK.m&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- Forward kinematics<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;qarm_IK.m&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- Inverse kinematics<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cubicTrajectory.m&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- Cubic spline segment<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;multiSegmentTrajectory.m&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- Multi-waypoint trajectory<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fruitDetector.m&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- Fruit detection<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pixelToWorld.m&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- Camera to world coords<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fruitSortingController.m&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- Main state machine<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;scripts/<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;create_FruitSorting_Model.m&nbsp;&nbsp;&nbsp;-- Simulink model generator<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;calibrate_camera.m&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- Camera calibration<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;validate_fruit_sorting.m&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- Full offline validation<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test_fruit_detection.m&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- Detection unit test<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;figures/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- Generated report figures<br/>"
        "</font>",
        s['CodeBlock']
    ))

    story.append(Paragraph("TASK ASSIGNMENT", s['SubHeader']))
    story.append(make_table(
        ["Team Member", "Assigned Tasks"],
        [
            ["Piero", "Build hardware Simulink model (Section 5), QUARC deployment"],
            ["Zihen", "Camera calibration (Section 6), tune HSV thresholds"],
            ["Ran", "Hardware testing + video recording (Section 9)"],
            ["Yichang", "Report writing (Section 10), presentation slides (Section 11)"],
            ["All", "Lab testing sessions, review report before submission"],
        ],
        col_widths=[35*mm, 125*mm]
    ))

    story.append(Paragraph("TIMELINE TO DEADLINE (1 May 2026)", s['SubHeader']))
    story.append(make_table(
        ["Week", "Tasks"],
        [
            ["Now - Week 1", "Build Simulink model, test on virtual QArm"],
            ["Week 2", "Camera calibration, tune detection, first hardware test"],
            ["Week 3", "Full autonomous + remote testing, record videos"],
            ["Week 4", "Write report, prepare slides, final testing"],
            ["Final week", "Review, submit report + peer assessment"],
        ],
        col_widths=[35*mm, 125*mm]
    ))

    # Build PDF
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print(f"PDF generated: {OUTPUT_PATH}")
    print(f"Total pages: {doc.page}")


if __name__ == "__main__":
    build_pdf()
