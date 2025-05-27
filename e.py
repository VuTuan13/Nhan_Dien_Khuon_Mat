import os
import shutil
from collections import Counter

def find_actors_with_100_images_1(datasets_path, output_path, num_actors=50):
    # Đường dẫn đến thư mục actor_faces và actress_faces
    actor_path = os.path.join(datasets_path, "actor_faces")
    actress_path = os.path.join(datasets_path, "actress_faces")
    
    selected_actors = []
    
    # Hàm để đếm số lượng ảnh trong mỗi thư mục
    def count_images_in_folders(base_path):
        actors_with_image_count = {}
        
        for actor_name in os.listdir(base_path):
            actor_folder = os.path.join(base_path, actor_name)
            if os.path.isdir(actor_folder):
                image_count = len([
                    f for f in os.listdir(actor_folder)
                    if os.path.isfile(os.path.join(actor_folder, f))
                ])
                actors_with_image_count[actor_name] = (base_path, image_count)
        
        return actors_with_image_count
    
    # Đếm số lượng ảnh cho mỗi diễn viên
    actor_counts = count_images_in_folders(actor_path)
    actress_counts = count_images_in_folders(actress_path)
    
    # Kết hợp cả hai dict
    all_actors = {**actor_counts, **actress_counts}
    
    # Lọc những người có >= 100 ảnh
    actors_with_100_images = {
        name: path_count for name, path_count in all_actors.items() if path_count[1] >= 100
    }
    
    # Lấy tối đa num_actors người
    actors_to_copy = list(actors_with_100_images.items())[:num_actors]
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Sao chép đúng 100 ảnh đầu tiên cho mỗi người
    for actor_name, (source_path, _) in actors_to_copy:
        source_actor_path = os.path.join(source_path, actor_name)
        dest_actor_path = os.path.join(output_path, actor_name)
        
        if not os.path.exists(dest_actor_path):
            os.makedirs(dest_actor_path)
        
        # Lấy danh sách ảnh và chỉ chọn 100 ảnh đầu tiên
        all_images = sorted([
            f for f in os.listdir(source_actor_path)
            if os.path.isfile(os.path.join(source_actor_path, f))
        ])[:100]
        
        for img in all_images:
            img_path = os.path.join(source_actor_path, img)
            shutil.copy2(img_path, os.path.join(dest_actor_path, img))
    
    print(f"Đã sao chép dữ liệu của {len(actors_to_copy)} diễn viên, mỗi người 100 ảnh.")
    return [name for name, _ in actors_to_copy]

selected = find_actors_with_100_images_1(
    datasets_path=r"D:\DoAnTotNghiep\Nhan_Dien_Khuon_Mat\datasets",  
    output_path=r"D:\DoAnTotNghiep\Nhan_Dien_Khuon_Mat\datasets\output\selected_actors_100_images",  
    num_actors=100
)

print("Danh sách diễn viên được chọn:")
for i, actor in enumerate(selected, 1):
    print(f"{i}. {actor}")
