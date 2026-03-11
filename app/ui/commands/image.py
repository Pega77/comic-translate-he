import numpy as np
import tempfile
from PySide6.QtGui import QUndoCommand
import imkit as imk


class SetImageCommand(QUndoCommand):
    def __init__(self, parent, file_path: str, img_array: np.ndarray, 
                 display: bool = True):
        super().__init__()
        self.ct = parent
        self.update_image_history(file_path, img_array)
        self.first = True
        self.display_first_time = display

    def redo(self):
        if self.first:
            if not self.display_first_time:
                return
            
            file_path = self.ct.image_files[self.ct.curr_img_idx]
            
            # Ensure the file has proper history initialization
            if file_path not in self.ct.current_history_index:
                self.ct.current_history_index[file_path] = 0
            if file_path not in self.ct.image_history:
                self.ct.image_history[file_path] = [file_path]
                
            current_index = self.ct.current_history_index[file_path]
            img_array = self.get_img(file_path, current_index)
            self.ct.image_viewer.display_image_array(img_array)
            self.first = False

        if self.ct.curr_img_idx >= 0:
            file_path = self.ct.image_files[self.ct.curr_img_idx]
            
            # Ensure proper initialization
            if file_path not in self.ct.current_history_index:
                self.ct.current_history_index[file_path] = 0
            if file_path not in self.ct.image_history:
                self.ct.image_history[file_path] = [file_path]
                
            current_index = self.ct.current_history_index[file_path]
            
            if current_index < len(self.ct.image_history[file_path]) - 1:
                current_index += 1
                self.ct.current_history_index[file_path] = current_index

                img_array = self.get_img(file_path, current_index)

                self.ct.image_data[file_path] = img_array
                self.ct.image_viewer.display_image_array(img_array)

    def undo(self):
        if self.ct.curr_img_idx >= 0:

            file_path = self.ct.image_files[self.ct.curr_img_idx]
            
            # Ensure proper initialization
            if file_path not in self.ct.current_history_index:
                self.ct.current_history_index[file_path] = 0
            if file_path not in self.ct.image_history:
                self.ct.image_history[file_path] = [file_path]
                
            current_index = self.ct.current_history_index[file_path]
            
            if current_index > 0:
                current_index -= 1
                self.ct.current_history_index[file_path] = current_index
                
                img_array = self.get_img(file_path, current_index)

                self.ct.image_data[file_path] = img_array
                self.ct.image_viewer.display_image_array(img_array)

   
    def update_image_history(self, file_path: str, img_array: np.ndarray):
        im = self.ct.load_image(file_path)

        if not np.array_equal(im, img_array):
            self.ct.image_data[file_path] = img_array
            
            # Update file path history
            history = self.ct.image_history[file_path]
            current_index = self.ct.current_history_index[file_path]
            
            # Remove any future history if we're not at the end
            del history[current_index + 1:]
            
            # # Save new image to temp file and add to history
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=self.ct.temp_dir)
            imk.write_image(temp_file.name, img_array)
            temp_file.close()

            history.append(temp_file.name)

            # Update in-memory history if this image is loaded
            if self.ct.in_memory_history.get(file_path, []):
                in_mem_history = self.ct.in_memory_history[file_path]
                del in_mem_history[current_index + 1:]
                in_mem_history.append(img_array.copy())

            self.ct.current_history_index[file_path] = len(history) - 1

    def get_img(self, file_path, current_index):
        if self.ct.in_memory_history.get(file_path, []):
            img_array = self.ct.in_memory_history[file_path][current_index]
        else:
            img_array = imk.read_image(self.ct.image_history[file_path][current_index])

        return img_array


class ToggleSkipImagesCommand(QUndoCommand):
    def __init__(self, main, file_paths: list[str], skip_status: bool):
        super().__init__()
        self.main = main
        self.file_paths = file_paths
        self.new_status = skip_status
        self.old_status = {
            path: main.image_states.get(path, {}).get('skip', False)
            for path in file_paths
        }

    def _apply_status(self, file_path: str, skip_status: bool):
        if file_path not in self.main.image_states:
            return
        self.main.image_states[file_path]['skip'] = skip_status

        try:
            idx = self.main.image_files.index(file_path)
        except ValueError:
            return

        item = self.main.page_list.item(idx)
        if item:
            fnt = item.font()
            fnt.setStrikeOut(skip_status)
            item.setFont(fnt)

        card = self.main.page_list.itemWidget(item) if item else None
        if card:
            card.set_skipped(skip_status)

    def redo(self):
        for file_path in self.file_paths:
            self._apply_status(file_path, self.new_status)

    def undo(self):
        for file_path in self.file_paths:
            self._apply_status(file_path, self.old_status.get(file_path, False))

class MirrorImageCommand(QUndoCommand):
    def __init__(self, main):
        super().__init__()
        self.ct = main
        self.prev_image = None
        self.prev_patches = None  # save patches for undo

    def redo(self):
        if self.ct.curr_img_idx < 0:
            return

        file_path = self.ct.image_files[self.ct.curr_img_idx]

        # 1. Composite ALL layers (base image + inpainted patches + text) into one flat image
        composited = self.ct.image_viewer.get_image_array(paint_all=True)
        if composited is None:
            return

        # 2. Save for undo
        self.prev_image = composited.copy()
        self.prev_patches = list(self.ct.image_patches.get(file_path, []))

        # 3. Clear stored patches — they are now baked into the flat image
        #    If we don't do this, load_patch_state() redraws them at original positions
        self.ct.image_patches[file_path] = []
        self.ct.in_memory_patches[file_path] = []

        # 4. Flip the flat composite horizontally
        mirrored = np.fliplr(composited).copy()

        # 5. Mirror blk_list bounding boxes
        img_w = composited.shape[1]
        if hasattr(self.ct, 'blk_list') and self.ct.blk_list:
            for blk in self.ct.blk_list:
                x1, y1, x2, y2 = blk.xyxy
                blk.xyxy = [img_w - x2, y1, img_w - x1, y2]

        # 6. Set the clean flipped image (no patches will be redrawn)
        self.ct.image_ctrl.set_image(mirrored)

    def undo(self):
        if self.prev_image is None or self.ct.curr_img_idx < 0:
            return

        file_path = self.ct.image_files[self.ct.curr_img_idx]

        # Restore original patches
        self.ct.image_patches[file_path] = self.prev_patches or []
        self.ct.in_memory_patches[file_path] = []

        # Un-flip blk_list
        img_w = self.prev_image.shape[1]
        if hasattr(self.ct, 'blk_list') and self.ct.blk_list:
            for blk in self.ct.blk_list:
                x1, y1, x2, y2 = blk.xyxy
                blk.xyxy = [img_w - x2, y1, img_w - x1, y2]

        self.ct.image_ctrl.set_image(self.prev_image.copy())
