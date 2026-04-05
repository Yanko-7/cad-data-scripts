from OCC.Extend.DataExchange import read_step_file
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.TopExp import topexp
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
import json
import os
import matplotlib.pyplot as plt

def is_watertight(shape):
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    for i in range(1, edge_face_map.Size() + 1):
        if edge_face_map.FindFromIndex(i).Size() < 2:
            return False
    return True

def analyze_step_file(step_file):
    try:
        shape = read_step_file(step_file)
        if not BRepCheck_Analyzer(shape).IsValid():
            return None
        if not is_watertight(shape):
            return None
        
        # Count faces and edges
        face_count = 0
        edge_count = 0
        
        # Count faces using TopExp_Explorer
        from OCC.Core.TopExp import TopExp_Explorer
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            face_count += 1
            face_explorer.Next()
        
        # Count edges using TopExp_Explorer
        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while edge_explorer.More():
            edge_count += 1
            edge_explorer.Next()
        
        return {
            'file': os.path.basename(step_file),
            'face_count': face_count,
            'edge_count': edge_count
        }
        
    except Exception as e:
        print(f"Error analyzing STEP file {os.path.basename(step_file)}: {e}")
        return None

def gather_step_files(directory):
    import os
    step_files = []
    for file in os.listdir(directory):
        if file.endswith(".step") or file.endswith(".stp"):
            step_files.append(os.path.join(directory, file))
    return step_files

def create_frequency_plots(face_freq, edge_freq, directory):
    """Create line plots for face and edge count frequency distributions"""
    
    # Set up the plot with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Face count plot
    face_counts = sorted(face_freq.items())
    face_x = [count for count, freq in face_counts]
    face_y = [freq for count, freq in face_counts]
    
    ax1.plot(face_x, face_y, marker='o', linewidth=2, markersize=4, color='blue')
    ax1.set_xlabel('Number of Faces')
    ax1.set_ylabel('Frequency (Number of Files)')
    ax1.set_title('Face Count Frequency Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Edge count plot
    edge_counts = sorted(edge_freq.items())
    edge_x = [count for count, freq in edge_counts]
    edge_y = [freq for count, freq in edge_counts]
    
    ax2.plot(edge_x, edge_y, marker='s', linewidth=2, markersize=4, color='red')
    ax2.set_xlabel('Number of Edges')
    ax2.set_ylabel('Frequency (Number of Files)')
    ax2.set_title('Edge Count Frequency Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_file = os.path.join(directory, 'step_frequency_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Frequency plots saved to: {plot_file}")
    
    # Also create a combined plot with both distributions
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(face_x, face_y, marker='o', linewidth=2, markersize=4, color='blue', label='Faces')
    plt.xlabel('Number of Faces')
    plt.ylabel('Frequency (Number of Files)')
    plt.title('Face Count Frequency Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(edge_x, edge_y, marker='s', linewidth=2, markersize=4, color='red', label='Edges')
    plt.xlabel('Number of Edges')
    plt.ylabel('Frequency (Number of Files)')
    plt.title('Edge Count Frequency Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    combined_plot_file = os.path.join(directory, 'step_frequency_combined.png')
    plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plots saved to: {combined_plot_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python stat_step.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    step_files = gather_step_files(directory)
    total = len(step_files)
    
    print(f"Found {total} STEP files to analyze...")
    
    # Analyze all files
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_step_file, step_file) for step_file in step_files]
        for future in tqdm(as_completed(futures), total=total, desc="Analyzing STEP files"):
            result = future.result()
            if result:
                results.append(result)
    
    print(f"Successfully analyzed {len(results)}/{total} STEP files")
    
    if not results:
        print("No valid files to analyze")
        sys.exit(1)
    
    # Extract face and edge counts
    face_counts = [r['face_count'] for r in results]
    edge_counts = [r['edge_count'] for r in results]
    
    # Calculate frequency statistics
    face_freq = Counter(face_counts)
    edge_freq = Counter(edge_counts)
    
    # Print statistics
    print("\n=== FACE COUNT STATISTICS ===")
    print(f"Total files with face data: {len(face_counts)}")
    print(f"Min faces: {min(face_counts)}")
    print(f"Max faces: {max(face_counts)}")
    print(f"Average faces: {sum(face_counts) / len(face_counts):.2f}")
    print(f"Median faces: {sorted(face_counts)[len(face_counts)//2]}")
    print("\nFace count frequency distribution:")
    for count, freq in sorted(face_freq.items()):
        print(f"  {count} faces: {freq} files ({freq/len(face_counts)*100:.1f}%)")
    
    print("\n=== EDGE COUNT STATISTICS ===")
    print(f"Total files with edge data: {len(edge_counts)}")
    print(f"Min edges: {min(edge_counts)}")
    print(f"Max edges: {max(edge_counts)}")
    print(f"Average edges: {sum(edge_counts) / len(edge_counts):.2f}")
    print(f"Median edges: {sorted(edge_counts)[len(edge_counts)//2]}")
    print("\nEdge count frequency distribution:")
    for count, freq in sorted(edge_freq.items()):
        print(f"  {count} edges: {freq} files ({freq/len(edge_counts)*100:.1f}%)")
    
    # Save detailed results to JSON
    output_data = {
        'summary': {
            'total_files_analyzed': len(results),
            'total_files_found': total,
            'face_stats': {
                'min': min(face_counts),
                'max': max(face_counts),
                'avg': sum(face_counts) / len(face_counts),
                'median': sorted(face_counts)[len(face_counts)//2]
            },
            'edge_stats': {
                'min': min(edge_counts),
                'max': max(edge_counts),
                'avg': sum(edge_counts) / len(edge_counts),
                'median': sorted(edge_counts)[len(edge_counts)//2]
            }
        },
        'frequency': {
            'face_counts': dict(face_freq),
            'edge_counts': dict(edge_freq)
        },
        'detailed_results': results
    }
    
    output_file = os.path.join(directory, 'step_statistics.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Create plots
    create_frequency_plots(face_freq, edge_freq, directory)

