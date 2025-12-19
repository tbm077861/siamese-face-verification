import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.filechooser import FileChooserListView
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from tensorflow.keras.layers import Layer

from kivy.uix.screenmanager import ScreenManager, Screen

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import shutil
from pathlib import Path

# Set the background color for the entire app
from kivy.core.window import Window
Window.clearcolor = (0.95, 0.95, 0.96, 1)  # Soft light gray-blue background

# siamese L1 Distance layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

class AuthScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app

        # Main layout with soft background
        main_layout = BoxLayout(
            orientation='vertical',
            padding=20,
            spacing=20
        )
        
        # Title with contrasting color
        title = Label(
            text='Face Authentication System',
            size_hint=(1, 0.2),
            font_size='24sp',
            bold=True,
            color=(0.2, 0.2, 0.3, 1)  # Dark blue-gray for contrast
        )
        
        # Camera widget
        self.camera = Camera(
            resolution=(640, 480),
            play=True,
            size_hint=(1, 0.6)
        )
        
        # Capture button with soft color
        self.capture_btn = Button(
            text='Capture & Authenticate',
            size_hint=(1, 0.1),
            background_color=(0.3, 0.5, 0.7, 1),  # Soft blue
            font_size='18sp',
            color=(1, 1, 1, 1)  # White text for contrast
        )
        self.capture_btn.bind(on_press=self.capture_and_authenticate)
        
        # Status label with contrasting color
        self.status_label = Label(
            text='Please position your face and click Capture',
            size_hint=(1, 0.1),
            font_size='16sp',
            color=(0.3, 0.3, 0.4, 1)  # Dark gray-blue
        )
        
        # Add widgets to layout
        main_layout.add_widget(title)
        main_layout.add_widget(self.camera)
        main_layout.add_widget(self.capture_btn)
        main_layout.add_widget(self.status_label)
        
        self.add_widget(main_layout)

    def capture_and_authenticate(self, instance):
        """Capture image from camera and authenticate"""
        self.status_label.text = "Processing..."
        self.capture_btn.disabled = True
        
        Clock.schedule_once(self.process_capture, 0.1)
    
    def process_capture(self, dt):
        """Process the captured image"""
        try:
            texture = self.camera.texture
            if texture is None:
                self.status_label.text = "Camera not ready. Please try again."
                self.capture_btn.disabled = False
                return
            
            buffer = texture.pixels
            image = np.frombuffer(buffer, dtype=np.uint8)
            image = image.reshape(texture.height, texture.width, 4)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            processed_image = self.app.preprocess_image(image)
            
            prediction = self.app.model.predict(
                [self.app.anchor_image, processed_image], 
                verbose=0
            )
            result = prediction[0][0]
            
            if result >= 0.5:
                self.app.show_success_popup()
            else:
                self.app.show_failure_popup()
                
        except Exception as e:
            Logger.error(f"FaceAuth: Error during authentication: {e}")
            self.status_label.text = "Authentication failed. Please try again."
        
        finally:
            self.capture_btn.disabled = False

class DashboardScreen(Screen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app

        # Enhanced dashboard with soft background
        main_layout = BoxLayout(
            orientation='vertical',
            padding=15,
            spacing=15
        )
        
        self.create_upload_directories()
        
        # Header with soft colors
        header_layout = BoxLayout(
            orientation='horizontal',
            size_hint=(1, 0.12),
            padding=10
        )
        
        title = Label(
            text='Welcome to Your Secure Dashboard',
            size_hint=(0.7, 1),
            font_size='26sp',
            bold=True,
            color=(0.2, 0.2, 0.3, 1)  # Dark contrasting color
        )
        
        logout_btn = Button(
            text='Logout',
            size_hint=(0.3, 0.8),
            background_color=(0.8, 0.4, 0.4, 1),  # Soft red
            font_size='16sp',
            bold=True,
            color=(1, 1, 1, 1)  # White text
        )
        logout_btn.bind(on_press=self.logout)
        
        header_layout.add_widget(title)
        header_layout.add_widget(logout_btn)
        main_layout.add_widget(header_layout)
        
        stats_layout = self.create_stats_section()
        main_layout.add_widget(stats_layout)
        
        content_layout = self.create_content_section()
        main_layout.add_widget(content_layout)
        
        self.load_files()
        self.add_widget(main_layout)
    
    def create_upload_directories(self):
        """Create directories for photos and documents"""
        self.photos_dir = Path("uploaded_photos")
        self.documents_dir = Path("uploaded_documents")
        
        self.photos_dir.mkdir(exist_ok=True)
        self.documents_dir.mkdir(exist_ok=True)
    
    def create_stats_section(self):
        """Create quick statistics section"""
        stats_layout = BoxLayout(
            orientation='horizontal', 
            size_hint=(1, 0.15), 
            spacing=15
        )
        
        # Photo count card with light background
        photo_card = BoxLayout(
            orientation='vertical',
            padding=10
        )
        # Add a light background to cards
        with photo_card.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.98, 0.98, 0.99, 1)  # Very light gray
            self.photo_rect = Rectangle(size=photo_card.size, pos=photo_card.pos)
        photo_card.bind(size=self._update_rect, pos=self._update_rect)
        
        photo_card.add_widget(Label(
            text='Photos',
            font_size='14sp',
            bold=True,
            color=(0.3, 0.3, 0.4, 1)  # Dark contrasting color
        ))
        self.photo_count = Label(
            text='0',
            font_size='24sp',
            bold=True,
            color=(0.2, 0.5, 0.7, 1)  # Blue accent
        )
        photo_card.add_widget(self.photo_count)
        
        # Document count card
        doc_card = BoxLayout(
            orientation='vertical',
            padding=10
        )
        with doc_card.canvas.before:
            Color(0.98, 0.98, 0.99, 1)
            self.doc_rect = Rectangle(size=doc_card.size, pos=doc_card.pos)
        doc_card.bind(size=self._update_rect, pos=self._update_rect)
        
        doc_card.add_widget(Label(
            text='Documents',
            font_size='14sp',
            bold=True,
            color=(0.3, 0.3, 0.4, 1)
        ))
        self.doc_count = Label(
            text='0',
            font_size='24sp',
            bold=True,
            color=(0.2, 0.6, 0.4, 1)  # Green accent
        )
        doc_card.add_widget(self.doc_count)
        
        # Total files card
        total_card = BoxLayout(
            orientation='vertical',
            padding=10
        )
        with total_card.canvas.before:
            Color(0.98, 0.98, 0.99, 1)
            self.total_rect = Rectangle(size=total_card.size, pos=total_card.pos)
        total_card.bind(size=self._update_rect, pos=self._update_rect)
        
        total_card.add_widget(Label(
            text='Total Files',
            font_size='14sp',
            bold=True,
            color=(0.3, 0.3, 0.4, 1)
        ))
        self.total_count = Label(
            text='0',
            font_size='24sp',
            bold=True,
            color=(0.7, 0.5, 0.2, 1)  # Orange accent
        )
        total_card.add_widget(self.total_count)
        
        stats_layout.add_widget(photo_card)
        stats_layout.add_widget(doc_card)
        stats_layout.add_widget(total_card)
        
        return stats_layout
    
    def _update_rect(self, instance, value):
        """Update card backgrounds"""
        instance.canvas.before.clear()
        with instance.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.98, 0.98, 0.99, 1)
            Rectangle(size=instance.size, pos=instance.pos)
    
    def create_content_section(self):
        """Create main content area with tabs"""
        content_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.73), spacing=10)
        
        # Action buttons with soft colors
        action_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=15)
        
        upload_photo_btn = Button(
            text='📷 Upload Photo',
            background_color=(0.3, 0.6, 0.4, 1),  # Soft green
            font_size='16sp',
            bold=True,
            color=(1, 1, 1, 1)  # White text
        )
        upload_photo_btn.bind(on_press=self.show_photo_upload)
        
        upload_doc_btn = Button(
            text='📄 Upload Document',
            background_color=(0.3, 0.5, 0.7, 1),  # Soft blue
            font_size='16sp',
            bold=True,
            color=(1, 1, 1, 1)  # White text
        )
        upload_doc_btn.bind(on_press=self.show_document_upload)
        
        refresh_btn = Button(
            text='🔄 Refresh',
            background_color=(0.7, 0.6, 0.3, 1),  # Soft orange
            font_size='16sp',
            bold=True,
            color=(1, 1, 1, 1)  # White text
        )
        refresh_btn.bind(on_press=self.refresh_files)
        
        action_layout.add_widget(upload_photo_btn)
        action_layout.add_widget(upload_doc_btn)
        action_layout.add_widget(refresh_btn)
        content_layout.add_widget(action_layout)
        
        # Tab-like sections
        tab_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.08), spacing=2)
        
        self.photos_tab = Button(
            text='Photo Gallery',
            background_color=(0.4, 0.6, 0.8, 1),  # Active tab color
            font_size='14sp',
            color=(1, 1, 1, 1)  # White text
        )
        self.photos_tab.bind(on_press=self.show_photos_tab)
        
        self.documents_tab = Button(
            text='Documents',
            background_color=(0.6, 0.6, 0.6, 1),  # Inactive tab color
            font_size='14sp',
            color=(1, 1, 1, 1)  # White text
        )
        self.documents_tab.bind(on_press=self.show_documents_tab)
        
        tab_layout.add_widget(self.photos_tab)
        tab_layout.add_widget(self.documents_tab)
        content_layout.add_widget(tab_layout)
        
        # Content display area
        self.content_display = BoxLayout(orientation='vertical', size_hint=(1, 0.82))
        
        # Photos section (default view)
        self.photos_scroll = ScrollView(size_hint=(1, 1))
        self.photos_grid = GridLayout(cols=4, spacing=15, size_hint_y=None, padding=10)
        self.photos_grid.bind(minimum_height=self.photos_grid.setter('height'))
        self.photos_scroll.add_widget(self.photos_grid)
        
        # Documents section (hidden initially)
        self.documents_scroll = ScrollView(size_hint=(1, 1))
        self.documents_layout = GridLayout(cols=1, spacing=10, size_hint_y=None, padding=10)
        self.documents_layout.bind(minimum_height=self.documents_layout.setter('height'))
        self.documents_scroll.add_widget(self.documents_layout)
        
        # Start with photos view
        self.content_display.add_widget(self.photos_scroll)
        self.current_tab = 'photos'
        
        content_layout.add_widget(self.content_display)
        
        return content_layout
    
    def show_photos_tab(self, instance):
        """Switch to photos tab"""
        if self.current_tab != 'photos':
            self.content_display.clear_widgets()
            self.content_display.add_widget(self.photos_scroll)
            self.photos_tab.background_color = (0.4, 0.6, 0.8, 1)  # Active blue
            self.documents_tab.background_color = (0.6, 0.6, 0.6, 1)  # Inactive gray
            self.current_tab = 'photos'
    
    def show_documents_tab(self, instance):
        """Switch to documents tab"""
        if self.current_tab != 'documents':
            self.content_display.clear_widgets()
            self.content_display.add_widget(self.documents_scroll)
            self.documents_tab.background_color = (0.4, 0.6, 0.8, 1)  # Active blue
            self.photos_tab.background_color = (0.6, 0.6, 0.6, 1)  # Inactive gray
            self.current_tab = 'documents'
    
    def logout(self, instance):
        """Logout and return to auth screen"""
        self.app.root.current = 'auth'
    
    def show_photo_upload(self, instance):
        """Show file chooser for photo upload"""
        content = BoxLayout(orientation='vertical', spacing=10)
        
        filechooser = FileChooserListView(
            filters=['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
        )
        content.add_widget(filechooser)
        
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), spacing=10)
        
        btn_upload = Button(
            text='Upload Photo', 
            background_color=(0.3, 0.6, 0.4, 1),
            color=(1, 1, 1, 1)
        )
        btn_cancel = Button(
            text='Cancel', 
            background_color=(0.8, 0.4, 0.4, 1),
            color=(1, 1, 1, 1)
        )
        
        button_layout.add_widget(btn_upload)
        button_layout.add_widget(btn_cancel)
        content.add_widget(button_layout)
        
        popup = Popup(
            title='Select Photo to Upload',
            content=content,
            size_hint=(0.9, 0.9),
            background_color=(0.95, 0.95, 0.96, 1)  # Match app background
        )
        
        def upload_photo(btn):
            if filechooser.selection:
                source_path = filechooser.selection[0]
                filename = os.path.basename(source_path)
                dest_path = self.photos_dir / filename
                
                try:
                    shutil.copy2(source_path, dest_path)
                    self.load_files()
                    popup.dismiss()
                    self.app.show_message("Photo uploaded successfully!")
                except Exception as e:
                    self.app.show_message(f"Error uploading photo: {e}")
        
        btn_upload.bind(on_press=upload_photo)
        btn_cancel.bind(on_press=popup.dismiss)
        
        popup.open()
    
    def show_document_upload(self, instance):
        """Show file chooser for document upload"""
        content = BoxLayout(orientation='vertical', spacing=10)
        
        filechooser = FileChooserListView(
            filters=['*.pdf', '*.doc', '*.docx', '*.txt', '*.rtf']
        )
        content.add_widget(filechooser)
        
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), spacing=10)
        
        btn_upload = Button(
            text='Upload Document', 
            background_color=(0.3, 0.5, 0.7, 1),
            color=(1, 1, 1, 1)
        )
        btn_cancel = Button(
            text='Cancel', 
            background_color=(0.8, 0.4, 0.4, 1),
            color=(1, 1, 1, 1)
        )
        
        button_layout.add_widget(btn_upload)
        button_layout.add_widget(btn_cancel)
        content.add_widget(button_layout)
        
        popup = Popup(
            title='Select Document to Upload',
            content=content,
            size_hint=(0.9, 0.9),
            background_color=(0.95, 0.95, 0.96, 1)  # Match app background
        )
        
        def upload_document(btn):
            if filechooser.selection:
                source_path = filechooser.selection[0]
                filename = os.path.basename(source_path)
                dest_path = self.documents_dir / filename
                
                try:
                    shutil.copy2(source_path, dest_path)
                    self.load_files()
                    popup.dismiss()
                    self.app.show_message("Document uploaded successfully!")
                except Exception as e:
                    self.app.show_message(f"Error uploading document: {e}")
        
        btn_upload.bind(on_press=upload_document)
        btn_cancel.bind(on_press=popup.dismiss)
        
        popup.open()
    
    def load_files(self):
        """Load and display uploaded files"""
        self.photos_grid.clear_widgets()
        self.documents_layout.clear_widgets()
        
        # Load photos
        photo_files = list(self.photos_dir.glob('*'))
        for photo_path in photo_files:
            if photo_path.is_file():
                try:
                    photo_card = BoxLayout(
                        orientation='vertical', 
                        size_hint_y=None, 
                        height=180, 
                        spacing=5,
                        padding=5
                    )
                    
                    # Add card background
                    with photo_card.canvas.before:
                        from kivy.graphics import Color, Rectangle
                        Color(0.98, 0.98, 0.99, 1)
                        Rectangle(size=photo_card.size, pos=photo_card.pos)
                    photo_card.bind(size=self._update_rect, pos=self._update_rect)
                    
                    photo_img = Image(
                        source=str(photo_path),
                        size_hint=(1, 0.7),
                        allow_stretch=True,
                        fit_mode="cover"
                    )
                    
                    info_layout = BoxLayout(
                        orientation='vertical',
                        size_hint=(1, 0.3),
                        spacing=2
                    )
                    
                    photo_name = Label(
                        text=photo_path.name[:20] + ('...' if len(photo_path.name) > 20 else ''),
                        size_hint=(1, 0.6),
                        text_size=(140, None),
                        halign='center',
                        font_size='12sp',
                        color=(0.3, 0.3, 0.4, 1)
                    )
                    
                    file_size = os.path.getsize(photo_path)
                    size_label = Label(
                        text=f"Size: {file_size/1024:.1f} KB",
                        size_hint=(1, 0.4),
                        text_size=(140, None),
                        halign='center',
                        font_size='10sp',
                        color=(0.5, 0.5, 0.6, 1)
                    )
                    
                    info_layout.add_widget(photo_name)
                    info_layout.add_widget(size_label)
                    
                    photo_card.add_widget(photo_img)
                    photo_card.add_widget(info_layout)
                    self.photos_grid.add_widget(photo_card)
                    
                except Exception as e:
                    Logger.error(f"Error loading photo {photo_path}: {e}")
        
        # Load documents
        doc_files = list(self.documents_dir.glob('*'))
        for doc_path in doc_files:
            if doc_path.is_file():
                doc_card = BoxLayout(
                    orientation='horizontal', 
                    size_hint_y=None, 
                    height=60, 
                    spacing=15,
                    padding=10
                )
                
                # Add card background
                with doc_card.canvas.before:
                    from kivy.graphics import Color, Rectangle
                    Color(0.98, 0.98, 0.99, 1)
                    Rectangle(size=doc_card.size, pos=doc_card.pos)
                doc_card.bind(size=self._update_rect, pos=self._update_rect)
                
                doc_icon = Label(
                    text='📄',
                    size_hint=(0.1, 1),
                    font_size='20sp'
                )
                
                doc_info = BoxLayout(orientation='vertical', size_hint=(0.7, 1))
                
                doc_name = Label(
                    text=doc_path.name,
                    size_hint=(1, 0.6),
                    halign='left',
                    text_size=(400, None),
                    bold=True,
                    color=(0.3, 0.3, 0.4, 1)
                )
                
                file_size = os.path.getsize(doc_path)
                doc_size = Label(
                    text=f"Size: {file_size/1024:.1f} KB",
                    size_hint=(1, 0.4),
                    halign='left',
                    text_size=(400, None),
                    font_size='12sp',
                    color=(0.5, 0.5, 0.6, 1)
                )
                
                doc_info.add_widget(doc_name)
                doc_info.add_widget(doc_size)
                
                action_layout = BoxLayout(orientation='horizontal', size_hint=(0.2, 1), spacing=5)
                
                doc_card.add_widget(doc_icon)
                doc_card.add_widget(doc_info)
                doc_card.add_widget(action_layout)
                self.documents_layout.add_widget(doc_card)
        
        # Update counts
        self.photo_count.text = str(len(photo_files))
        self.doc_count.text = str(len(doc_files))
        self.total_count.text = str(len(photo_files) + len(doc_files))
        
        # Update grid heights
        self.photos_grid.height = max(200, (len(photo_files) + 3) // 4 * 190)
        self.documents_layout.height = max(70, len(doc_files) * 70)
    
    def refresh_files(self, instance):
        """Refresh the file display"""
        self.load_files()
        self.app.show_message("Files refreshed!")

class FaceAuthApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.anchor_image = None
        
    def build(self):
        ANCHOR_PATH = 'data/anchor/anchor.jpg'
        try:
            self.model = load_model(
                'siamesemodel.h5',
                custom_objects={'L1Dist': L1Dist}
            )
            Logger.info("FaceAuth: Model loaded successfully")

            if not os.path.exists(ANCHOR_PATH):
                msg = f"Anchor image not found at: {ANCHOR_PATH}"
                Logger.error(f"FaceAuth: {msg}")
                self.show_error_popup(msg)
                return Label(text=msg)
            
            anchor_img_raw = cv2.imread(ANCHOR_PATH)
            if anchor_img_raw is None:
                msg = f"Failed to read anchor image from: {ANCHOR_PATH}"
                Logger.error(f"FaceAuth: {msg}")
                self.show_error_popup(msg)
                return Label(text=msg)

            self.anchor_image = self.preprocess_image(anchor_img_raw)
            Logger.info("FaceAuth: Anchor image loaded and preprocessed.")
        except Exception as e:
            Logger.error(f"FaceAuth: Error loading model: {e}")
            self.show_error_popup("Model loading failed")
            return Label(text="Model loading failed")
        
        sm = ScreenManager()
        sm.add_widget(AuthScreen(app=self, name='auth'))
        sm.add_widget(DashboardScreen(app=self, name='dashboard'))
        sm.current = 'auth'
        
        return sm
    
    def show_success_popup(self):
        """Show success popup and transition to dashboard"""
        content = BoxLayout(orientation='vertical', padding=20, spacing=20)
        content.add_widget(Label(
            text='Authentication Successful!',
            font_size='20sp',
            color=(0, 0.5, 0, 1)  # Dark green for contrast
        ))
        
        btn = Button(
            text='Continue to Dashboard',
            size_hint=(1, 0.4),
            background_color=(0.3, 0.6, 0.4, 1),
            color=(1, 1, 1, 1)
        )
        content.add_widget(btn)
        
        popup = Popup(
            title='Success',
            content=content,
            size_hint=(0.7, 0.4),
            auto_dismiss=False,
            background_color=(0.95, 0.95, 0.96, 1)  # Match app background
        )
        
        def go_to_dashboard(instance):
            popup.dismiss()
            self.root.current = 'dashboard'
        
        btn.bind(on_press=go_to_dashboard)
        popup.open()
        self.root.get_screen('auth').status_label.text = "Authentication successful!"
    
    def show_failure_popup(self):
        """Show failure popup"""
        content = BoxLayout(orientation='vertical', padding=20, spacing=20)
        content.add_widget(Label(
            text='Authentication Failed!',
            font_size='20sp',
            color=(0.7, 0.2, 0.2, 1)  # Dark red for contrast
        ))
        content.add_widget(Label(
            text='Please try again or contact administrator.',
            font_size='16sp',
            color=(0.3, 0.3, 0.4, 1)
        ))
        
        btn = Button(
            text='Try Again',
            size_hint=(1, 0.4),
            background_color=(0.8, 0.4, 0.4, 1),
            color=(1, 1, 1, 1)
        )
        content.add_widget(btn)
        
        popup = Popup(
            title='Access Denied',
            content=content,
            size_hint=(0.7, 0.5),
            auto_dismiss=False,
            background_color=(0.95, 0.95, 0.96, 1)  # Match app background
        )
        
        btn.bind(on_press=popup.dismiss)
        popup.open()
        self.root.get_screen('auth').status_label.text = "Authentication failed. Please try again."
        
    def preprocess_image(self, image):
        """Preprocess the image for your SNN model"""
        try:
            resized = cv2.resize(image, (100, 100))
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb_image.astype('float32') / 255.0
            batched = np.expand_dims(normalized, axis=0)
            return batched
            
        except Exception as e:
            Logger.error(f"FaceAuth: Error preprocessing image: {e}")
            raise
    
    def show_error_popup(self, message):
        """Show error popup"""
        content = BoxLayout(orientation='vertical', padding=20, spacing=20)
        content.add_widget(Label(
            text=message, 
            font_size='16sp',
            color=(0.3, 0.3, 0.4, 1)
        ))
        
        btn = Button(
            text='OK', 
            size_hint=(1, 0.4),
            background_color=(0.3, 0.5, 0.7, 1),
            color=(1, 1, 1, 1)
        )
        content.add_widget(btn)
        
        popup = Popup(
            title='Error',
            content=content,
            size_hint=(0.6, 0.4),
            background_color=(0.95, 0.95, 0.96, 1)  # Match app background
        )
        
        btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def show_message(self, message):
        """Show a simple message popup"""
        content = BoxLayout(orientation='vertical', padding=20, spacing=20)
        content.add_widget(Label(
            text=message, 
            font_size='16sp',
            color=(0.3, 0.3, 0.4, 1)
        ))
        
        btn = Button(
            text='OK', 
            size_hint=(1, 0.4),
            background_color=(0.3, 0.5, 0.7, 1),
            color=(1, 1, 1, 1)
        )
        content.add_widget(btn)
        
        popup = Popup(
            title='Info',
            content=content,
            size_hint=(0.6, 0.3),
            background_color=(0.95, 0.95, 0.96, 1)  # Match app background
        )
        
        btn.bind(on_press=popup.dismiss)
        popup.open()

if __name__ == '__main__':
    FaceAuthApp().run()