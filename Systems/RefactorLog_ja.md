# Refactoring Proposal for Systems Directory

## 現在のコードアーキテクチャ

現在のSystemsディレクトリには以下のクラスが含まれています：

1. **BaseSystem** - メインの実行フレームワーク、データロード、オクルージョンマスク生成、レンダリングを調整
2. **DataLoader** - RGB画像、深度画像、IMUデータのロード責任
3. **ModelLoader** - 3Dモデルとシーン記述のロード責任
4. **Renderer** - オクルージョンマスクとシーン記述に基づいてMRシーンをレンダリング
5. **ContentsDepthCal** - MRコンテンツの深度マップを計算
6. **VisualizeSystem** - MRシーンを3Dで可視化するためのシステム
7. **VisualizeModelRender** - 可視化のためのモダンOpenGLベースのレンダリングシステム

### 依存関係 

```
BaseSystem
 ├── DataLoader
 ├── ModelLoader
 ├── Renderer
 └── ContentsDepthCal
      └── Renderer

VisualizeSystem
 ├── DataLoader
 ├── ModelLoader
 └── VisualizeModelRender
```

### 問題点 

1. **OpenGLの重複実装** - `Renderer`と`VisualizeModelRender`の間でOpenGLの初期化と利用に重複がある
2. **レンダリングコードの混在** - `Renderer`、`ContentsDepthCal`、`VisualizeModelRender`で類似したレンダリングコードが複数存在
3. **Transform処理の重複** - 各クラスに同様の変換（position、rotation、scale）処理コードがある
4. **異なるOpenGLアプローチ** - レガシーOpenGL（即時モード）とモダンOpenGL（シェーダー）が混在
5. **モデルローディングの重複** - 複数の場所でモデルローディングロジックが実装されている
6. **モジュールの責任範囲が不明瞭** - いくつかのクラスで責任の境界が曖昧

## リファクタリング提案 (Refactoring Proposal)

### 1. レンダリングサブシステムの統合

`Renderer`と`VisualizeModelRender`を単一の統合レンダリングシステムに再構築します。

```
Rendering/
 ├── Renderer.py(Interface)               - 基本レンダリングインターフェース
 ├── OpenGLRenderer.py         - OpenGL実装（モダンアプローチ）, extends Renderer
 ├── ShaderManager.py          - シェーダープログラムの管理
 ├── GeometryPrimitives.py     - 基本的な幾何形状（キューブ、グリッド、矢印など）
 ├── TextureManager.py         - テクスチャ読み込みと管理
 └── ModelRenderer.py          - 3Dモデルのレンダリングロジック
```

### 2. モデル管理の改善

モデル読み込みと管理を一貫したインターフェースに統合します。

```
Models/
 ├── ModelLoader.py            - 統合モデルローダーインターフェース
 ├── ObjModelLoader.py         - OBJファイル用ローダー extends ModelLoader
 ├── FbxModelLoader.py         - FBXファイル用ローダー extends ModelLoader
 ├── Model.py                 - 統一モデル表現
 ├── SceneManager.py           - シーン構成の管理
 └── TransformUtils.py         - 共通の変換ユーティリティ
```

### 3. システムアーキテクチャの改善

```
Systems/
 ├── BaseSystem.py - facade for Occlusion System and VisualizeSystem
 ├── OcclusionSystem.py             - Currnent Base System
 ├── VisualizeSystem.py        - 可視化システム（新しいレンダリングシステムを使用）
 └── OcclusionProcessor.py     - オクルージョン処理（ContentsDepthCalの責任を含む）
```

## 4. DataLoader Architecture

```
DataLoader

 ├── Frame.py - Curernt Frame.py in Interfaces
 ├── UniformedFrameLoader.py      - Current Frame loader (adopt before DepthIMUData2)
 ├── SepeartedFrameLoader.py      - New filesystem Frame loader (adopt after DepthIMUData3)
 └── JsonBasedSceneLoader.py - Current Scene Loader
```

### 4. インターフェースの明確化

```
core/
 ├── IRenderer.py     - レンダリングシステムの抽象インターフェース
 ├── IModel.py         - モデル管理の抽象インターフェース
 ├── IFrameLoader.py  - the Frame loader framework
 ├── ISceneLoader.py  - Scene loader framework
 ├── ITracker.py - Abstraction of Tracker(current Tracker.py)
 ├── IOcclusionProvider.py - Abstraction of Tracker(current OcclusionProvider.py)
 └── IScene.py - Abstraction of MR Scene
 
```

## 具体的な改善点 (Specific Improvements)

### 1. OpenGL初期化の共通化

`OpenGLRenderer`クラスにOpenGL初期化を集約し、レガシーとモダンの両方のレンダリングパスをサポート：

```python
class OpenGLRenderer:
    def __init__(self, use_modern=True):
        self.use_modern = use_modern
        self._initialize_opengl()
        
    def _initialize_opengl(self):
        # 共通初期化
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        
        if self.use_modern:
            # モダンOpenGLの初期化（シェーダーベース）
            # ...
        else:
            # レガシーOpenGLの初期化
            # ...
```

### 2. 変換ユーティリティの統合

位置、回転、スケールなどの変換処理を集約：

```python
class TransformUtils:
    @staticmethod
    def create_transform_matrix(position, rotation, scale=None):
        """
        Create a combined transformation matrix from position, rotation and scale.
        
        Args:
            position: Dict with x, y, z position values
            rotation: Dict with x, y, z, w quaternion or x, y, z Euler angles
            scale: Optional Dict with x, y, z scale values
            
        Returns:
            4x4 transformation matrix as numpy array
        """
        # Implementation...
        
    @staticmethod
    def quaternion_from_euler(euler_angles):
        """Convert Euler angles to quaternion"""
        # Implementation...
        
    @staticmethod
    def euler_from_quaternion(quaternion):
        """Convert quaternion to Euler angles"""
        # Implementation...
```

### 3. モデル読み込みの統合

モデル読み込みロジックの統合：

```python
class ModelLoader:
    def __init__(self):
        self.loaders = {
            '.obj': ObjModelLoader(),
            '.fbx': FbxModelLoader()
        }
        self.model_cache = {}
        
    def load_model(self, file_path):
        """Load a 3D model with appropriate loader based on extension"""
        if file_path in self.model_cache:
            return self.model_cache[file_path]
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.loaders:
            raise ValueError(f"Unsupported model format: {ext}")
            
        model = self.loaders[ext].load(file_path)
        self.model_cache[file_path] = model
        return model
```

### 4. データパイプラインの改良

並列処理やメモリ効率の改善：

```python
class DataPipeline:
    def __init__(self, data_dirs, batch_size=1, preload=False):
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.preload = preload
        self._frames = {}
        
    def load_data(self, on_progress=None):
        """Load data with optional parallel processing"""
        # Implementation with parallel processing option
        
    def get_frame_batch(self, start_idx, count=None):
        """Get a batch of frames for processing"""
        # Implementation...
```

### 5. シェーダー管理の改善

```python
class ShaderManager:
    def __init__(self):
        self.shaders = {}
        
    def load_shader(self, name, vertex_source, fragment_source):
        """Compile and store a shader program"""
        # Implementation...
        
    def use_shader(self, name):
        """Activate a specific shader program"""
        # Implementation...
        
    def set_uniform(self, name, uniform_name, value):
        """Set a uniform value for a shader"""
        # Implementation...
```

## 実装ステップ (Implementation Steps)

1. 共通のインターフェースとユーティリティクラス (`TransformUtils`など) を最初に作成
2. 新しいレンダリングシステム (`Rendering/` ディレクトリ) を構築
3. モデル管理システム (`Models/` ディレクトリ) を実装
4. データパイプラインを改良
5. `BaseSystem`と`VisualizeSystem`を新しいコンポーネントを使用するように更新
6. ユニットテストを作成して新しいアーキテクチャの堅牢性を確保

## 期待される利点 (Expected Benefits)

1. **コードの重複削減** - 共通のユーティリティとインターフェースによる改善
2. **メンテナンス性の向上** - 明確な責任分離
3. **拡張性の向上** - 新しいレンダラーやファイル形式の追加が容易
4. **パフォーマンス向上** - モダンOpenGLの統一的な利用
5. **テスト容易性** - インターフェースに基づく設計によるユニットテスト向上

## まとめ (Summary)

このリファクタリング提案は、現在のフレームワークの基本的な機能を維持しながら、コードの重複を減らし、関心事の分離を改善し、より高いモジュール性と拡張性を提供することを目指しています。最新のソフトウェア設計原則に従い、将来的にさらなる機能拡張を行う際により堅牢な基盤になるでしょう。