import fitz
from PIL import Image
from typing import List, Tuple, Union, Any

class PDFReader:
    # 9.9 v0.1(add reset_cursor, drag)
    # 9.14 v0.2(add padding, split, chunk_page) 
    # 9.18 add __len__
    # 10.14 add __geitem__
    def load(self, file_path: str) -> None:
        self.doc = fitz.open(file_path)
        
    def get_text(self, page_num: int) -> str:
        """
        Arguments:
            page_num(int): page number of the target page(from 0 to len - 1)
        Returns:
            str: Text on page `page_num`
        """
        assert hasattr(self, 'doc') and page_num < len(self.doc)
        return self.doc.load_page(page_num).get_text()
    def get_all_text(self) -> List[str]:
        assert hasattr(self, 'doc')
        return [self.get_text(page_num) for page_num in range(len(self.doc))]
    
    def capture(self,
                page_num: int,
                path: str,
                area: Union[Tuple[float, float, float, float], None]=None,
                dpi: int=None,
                return_image: bool=False
                ) -> Union[None, Image.Image]:
        """
        Capture an image from the target page.
        Arguments:
            page_num(int): page number of the target page.
            path(str): image file path.
            area(Tuple[float, float, float, float] | None): target area in the xyxy format.
        """
        page = self.doc.load_page(page_num)
        if area is not None:
            page_width, page_height = page.rect.width, page.rect.height
            rect = fitz.Rect(area[0] * page_width, area[1] * page_height, area[2] * page_width, area[3] * page_height)
        else:
            rect = None
        pix = page.get_pixmap(clip=rect, dpi=dpi)
        pix.save(path)

        if return_image:
            return Image.open(path)
    
    def padding(self, image: Image.Image, divider: Tuple[int, int], type='center') -> Image.Image:
        w, h = image.size
        new_width = ((w - 1) // divider[0] + 1) * divider[0]
        new_height = ((h - 1) // divider[1] + 1) * divider[1]

        new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        new_image.paste(image, ((new_width - w) // 2, (new_height - h) // 2))

        return new_image
    
    def split(self, image: Image.Image, w_split: int, h_split: int, order='h') -> List[Image.Image]:
        assert order in ['h', 'w']
        padded_image = self.padding(image, (w_split, h_split))
        block_w, block_h = padded_image.size[0] // w_split, padded_image.size[1] // h_split
        blocks = []
        if order == 'h':
            for i in range(w_split):
                for j in range(h_split):
                    block = image.crop((i * block_w, j * block_h, (i + 1) * block_w, (j + 1) * block_h))
                    blocks.append(block)
        elif order == 'w':
            for j in range(h_split):
                for i in range(w_split):
                    block = image.crop((i * block_w, j * block_h, (i + 1) * block_w, (j + 1) * block_h))
                    blocks.append(block)
        
        return blocks
    
    def reset_cursor(self, page_num: int, path: str):
        self.cursor = [0.0, 0.0, 1.0, 1.0]
        self.page_num = page_num
        self.capture(page_num, path, area=tuple(self.cursor), dpi=100)
    
    def drag(self, action: str, path: str):
        """
        action: str such as 'up_up', 'up_down', 'left_left', ...
        """
        assert hasattr(self, 'cursor')
        side, direction = action.split('_')
        self.pre_cursor = self.cursor.copy()
        index = ['left', 'up', 'right', 'down'].index(side)
        boundary = [(0.0, self.pre_cursor[2]), (self.pre_cursor[0], 1.0), (0.0, self.pre_cursor[3]), (self.pre_cursor[1], 1.0)][index]
        if direction in ['left', 'up']:
            self.cursor[index] = (self.cursor[index] + boundary[0]) / 2
        else:
            self.cursor[index] = (self.cursor[index] + boundary[1]) / 2
        
        self.capture(self.page_num, path, area=tuple(self.cursor), dpi=100)
    
    def chunk_page(self, page_num: int, w_split: int=2, h_split: int=2, dpi: int=None,
                         overview_only: bool=False,
                         image_only: bool=False) -> List[Union[Image.Image, str]]:
        text = self.get_text(page_num)
        capture = self.capture(page_num, dpi=None, path='tmp.png', return_image=True)
        if overview_only:
            blocks = []
        else:
            blocks = self.split(capture, w_split, h_split, order='w')
        if image_only:
            return [*blocks, capture]

        return [text, *blocks, capture]
    
    def __len__(self):
        return self.doc.page_count

if __name__ == '__main__':
    reader = PDFReader()
    reader.load("./data/2410.04790v1.pdf")
    for i in range(len(reader)):
        reader.capture(i, f"data/page_{i}.png", dpi=144)