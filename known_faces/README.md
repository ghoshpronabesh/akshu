# Known Faces Directory

This directory contains reference images for face recognition.

## Structure
```
known_faces/
├── person_name/          # Create a folder for each person
│   ├── image1.jpg        # Add multiple photos of the person
│   ├── image2.jpg        # Different angles and lighting
│   └── image3.jpg        # Improves recognition accuracy
└── encodings.pkl         # Cached face encodings (auto-generated)
```

## Instructions

1. **Adding a New Person:**
   - Create a folder with the person's name (e.g., `john_doe`)
   - Add 2-5 clear photos of the person's face
   - Supported formats: .jpg, .jpeg, .png
   - Ensure good lighting and clear face visibility

2. **Image Guidelines:**
   - Face should be clearly visible
   - Good lighting conditions
   - Different angles for better accuracy
   - Image resolution: at least 200x200 pixels
   - One face per image (primary face will be used if multiple)

3. **Folder Naming:**
   - Use underscores for spaces (e.g., `john_doe` instead of `john doe`)
   - Avoid special characters
   - Use lowercase for consistency

## Example Setup
```
known_faces/
├── john_doe/
│   ├── john_front.jpg
│   ├── john_side.jpg
│   └── john_smile.jpg
├── jane_smith/
│   ├── jane_1.jpg
│   └── jane_2.jpg
└── encodings.pkl
```

The system will automatically process these images and create face encodings for recognition.