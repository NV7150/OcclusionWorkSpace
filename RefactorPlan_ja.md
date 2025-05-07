# MRオクルージョンフレームワークのリファクタリング計画書

## 1. はじめに

本文書は、MRオクルージョンフレームワークのための改良されたリファクタリング計画の概要を示します。これは、オリジナルの`Systems/RefactorLog.md`で提示された優れた分析と提案に基づいています。

このリファクタリング作業の主な目的は以下の通りです。
*   **コードの重複排除:** 特にレンダリング、モデル読み込み、変換ロジックにおける冗長なコードを排除します。
*   **モジュール性の向上:** 明確な責任を持つ、適切に定義された独立したモジュールを作成します。
*   **関心の明確な分離:** 各コンポーネントがシステムの特定の側面に集中するようにします。
*   **モダンOpenGLプラクティスの採用:** すべてのレンダリングタスクに対して、シェーダーベースのモダンOpenGLパイプラインに移行します。
*   **保守性と拡張性の強化:** コードベースを理解しやすく、修正しやすく、新しい機能やアルゴリズムで拡張しやすくします。

## 2. 対処される主要な問題点

このリファクタリングは、現在のコードベースで特定された以下の主要な問題に対処します。
*   **OpenGL実装の重複:** 複数の異なるOpenGL初期化とレンダリングロジック。
*   **混在したレンダリングコード:** 同様のレンダリングタスクがモジュール間で異なる方法で実装されている。
*   **変換処理の重複:** 変換ロジック（位置、回転、スケール）がいくつかのクラスで繰り返されている。
*   **異なるOpenGLアプローチ:** レガシー（イミディエイトモード）とモダン（シェーダーベース）OpenGLの共存。
*   **モデル読み込みの重複:** モデルデータの読み込みと解析ロジックが異なるコンポーネントに分散している。
*   **不明確なモジュールの責任:** 一部のクラスの境界が曖昧で、依存関係が複雑になっている。

## 3. 提案される新しいアーキテクチャ

### 3.1. 全体的なディレクトリ構造

提案される新しいディレクトリ構造は以下の通りです。

```
MR_Occlusion_Framework/
├── core/                      # コアインターフェース (抽象基底クラス)
│   ├── IFrameLoader.py
│   ├── IOcclusionProvider.py
│   ├── IModel.py
│   ├── IRenderer.py
│   ├── IScene.py
│   └── ITracker.py
├── DataLoaders/               # データ読み込み (フレーム、センサーデータ)
│   ├── Frame.py               # (Interfaces/ から移動)
│   ├── BaseFrameLoader.py     # (オプション: 異なるフレームソースのためのABC)
│   ├── UniformedFrameLoader.py # (現在の Systems/DataLoader.py のロジック)
│   └── SeparatedFrameLoader.py # (将来の DepthIMUData3 構造のため)
├── Models/                    # 3Dモデルとシーンの表現/管理
│   ├── Model.py               # 統一された3Dモデルデータ構造 (頂点、法線、UV、マテリアルなど)
│   ├── Mesh.py                # Modelの個々のメッシュコンポーネント
│   ├── Material.py            # マテリアルプロパティ
│   ├── Texture.py             # テクスチャデータ
│   ├── BaseSceneLoader.py     # シーン記述ローダーのためのABC
│   ├── JsonSceneLoader.py     # JSONからシーン構造を読み込む (現在の Systems/ModelLoader.py のJSONロジック)
│   ├── SceneManager.py        # シーングラフ、オブジェクトインスタンス、およびそれらの変換を管理
│   ├── BaseAssetLoader.py     # 特定のモデルファイル形式ローダーのためのABC
│   ├── FbxLoader.py           # FBXファイルをModel/Mesh構造に読み込む
│   └── ObjLoader.py           # OBJファイルをModel/Mesh構造に読み込む
├── Rendering/                 # 統一されたモダンOpenGLレンダリングサブシステム
│   ├── Renderer.py            # メインレンダリングクラス (モダンOpenGL、シェーダーベース)
│   ├── ShaderManager.py       # GLSLシェーダープログラムを管理
│   ├── TextureManager.py      # OpenGLテクスチャを管理
│   ├── BufferManager.py       # VBO、EBO、VAOを管理
│   ├── Framebuffer.py         # フレームバッファオブジェクト (FBO) 管理
│   ├── Camera.py              # ビュー/プロジェクション行列のためのカメラクラス
│   └── Primitives.py          # 基本形状 (グリッド、軸、デバッグビジュアル) の描画用
├── Systems/                   # 高レベルシステムコーディネーター
│   ├── OcclusionSystem.py     # (現在の Systems/BaseSystem.py からリファクタリング)
│   ├── VisualizeSystem.py     # (新しい Rendering/ および Models/ サブシステムを使用するようにリファクタリング)
│   └── OcclusionProcessor.py  # オクルージョンロジックを処理し、MRコンテンツ深度のためにRendererを使用
├── Trackers/                  # トラッキング実装
│   └── ApriltagTracker.py     # (Tracker/ から移動)
├── OcclusionProviders/        # オクルージョンアルゴリズム実装
│   ├── DepthThresholdOcclusion.py # (例、Occlusions/ から移動)
│   └── SimpleOcclusion.py     # (例、Occlusions/ から移動)
└── Utils/                     # 共通ユーティリティ
    ├── TransformUtils.py      # 行列演算、クォータニオン/オイラー角などのため
    ├── Logger.py              # (Logger/ から移動)
    └── MarkerPositionLoader.py # (変更なし、またはSceneAssetタイプに統合)
```

### 3.2. コンポーネントの責任

*   **`core/`**: 主要コンポーネントの抽象基底クラス（インターフェース）を定義します。これにより、ポリモーフィズムが促進され、異なる実装を容易に交換できます。
*   **`DataLoaders/`**: すべての時系列センサーデータ（RGB、深度、IMU）の読み込みを処理し、`Frame`オブジェクトを構築します。`Frame.py`は、タイムスタンプごとのデータの標準化されたデータ構造になります。
*   **`Models/`**: すべての3Dアセット（モデル、メッシュ、マテリアル、テクスチャ）の読み込み、表現、管理、およびシーングラフ情報を一元化します。
    *   `Model.py` & `Mesh.py`: 3Dジオメトリのメモリ内表現を定義します。
    *   `FbxLoader.py` & `ObjLoader.py`: 特定のファイル形式を`Model/Mesh`構造に解析するための具体的な実装。
    *   `JsonSceneLoader.py`: シーン記述ファイル（オブジェクトの配置、モデルIDなど）を解析します。
    *   `SceneManager.py`: シーン内のモデルインスタンスのコレクション、それらの変換、および関係を管理します。
*   **`Rendering/`**: 単一の、モダンな、シェーダーベースのOpenGLレンダリングエンジン。
    *   `Renderer.py`: レンダリングパスを調整するメインクラス。画面へのレンダリング（視覚化用）とオフスクリーンフレームバッファへのレンダリング（深度計算、最終合成用）をサポートします。
    *   `ShaderManager.py`, `TextureManager.py`, `BufferManager.py`: OpenGLリソースを効率的に管理します。
    *   `Camera.py`: ビューおよびプロジェクション行列の計算を処理します。
    *   `Primitives.py`: デバッグまたは視覚化支援のための単純な幾何学的形状を描画するためのユーティリティ。
    *   現在の`ContentsDepthCal.py`の機能は、この`Renderer.py`の特定のレンダリングパスまたは機能として統合されます。
*   **`Systems/`**: 全体的なワークフローを調整する高レベルクラス。
    *   `OcclusionSystem.py`: オクルージョン処理パイプラインのメインエントリポイント。
    *   `VisualizeSystem.py`: 3D視覚化のメインエントリポイント。
    *   `OcclusionProcessor.py`: 仮想コンテンツの深度マップを取得するために`Rendering/`サブシステムを利用して、オクルージョンマスクを生成するロジックをカプセル化します。
*   **`Trackers/` & `OcclusionProviders/`**: それぞれ`ITracker`および`IOcclusionProvider`インターフェースの具体的な実装を含みます。
*   **`Utils/`**: 数学演算のための`TransformUtils.py`や既存の`Logger.py`などの共有ユーティリティモジュールを格納します。

## 4. 主な変更点と利点

*   **統一されたモダンOpenGLパイプライン:** `Rendering/`内の単一で一貫性のあるシェーダーベースのレンダリングエンジンが、複数の混合アプローチのOpenGL実装を置き換えます。これにより、パフォーマンス、保守性が向上し、最新のグラフィックス機能を活用できます。
*   **一元化されたモデルとシーン管理:** `Models/`サブシステムは、3Dアセットとシーン構造を読み込み、表現、管理するための明確で統一された方法を提供し、冗長性を排除します。
*   **統合された変換ロジック:** `Utils/TransformUtils.py`は、すべての3D変換数学のための一元的な場所を提供し、一貫性を確保し、エラーを削減します。
*   **より明確なモジュール境界:** 新しいディレクトリとクラス構造により、関心の分離が向上し、システムが理解しやすく、変更しやすくなります。
*   **テスト容易性の向上:** 明確に定義されたインターフェースとモジュール化されたコンポーネントにより、単体テストが容易になります。
*   **拡張性の強化:** モジュール設計と明確なインターフェースにより、新しいモデル形式、レンダリング技術、またはデータソースの追加が簡単になります。

## 5. Mermaid図 (クラス関連図)

```mermaid
classDiagram
    direction LR

    package Systems {
        class OcclusionSystem
        class VisualizeSystem
        class OcclusionProcessor
    }

    package Rendering {
        class Renderer
        class ShaderManager
        class TextureManager
        class BufferManager
        class Framebuffer
        class Camera
        class Primitives
    }

    package Models {
        class Model
        class Mesh
        class Material
        class Texture
        class JsonSceneLoader
        class SceneManager
        class FbxLoader
        class ObjLoader
        class BaseAssetLoader
        class BaseSceneLoader
    }

    package DataLoaders {
        class Frame
        class UniformedFrameLoader
        class SeparatedFrameLoader
        class BaseFrameLoader
    }

    package core {
        class IRenderer <<Interface>>
        class IModel <<Interface>>
        class IScene <<Interface>>
        class IFrameLoader <<Interface>>
        class IOcclusionProvider <<Interface>>
        class ITracker <<Interface>>
    }
    
    package Utils {
        class TransformUtils
        class Logger
        class MarkerPositionLoader
    }

    package Trackers {
        class ApriltagTracker
    }

    package OcclusionProviders {
        class ConcreteOcclusionProvider
    }

    OcclusionSystem ..> IOcclusionProvider
    OcclusionSystem ..> IFrameLoader
    OcclusionSystem ..> IScene
    OcclusionSystem ..> IRenderer
    OcclusionSystem ..> OcclusionProcessor
    OcclusionProcessor ..> IRenderer
    OcclusionProcessor ..> IModel

    VisualizeSystem ..> IFrameLoader
    VisualizeSystem ..> IScene
    VisualizeSystem ..> IRenderer
    VisualizeSystem ..> ITracker

    Rendering.Renderer ..> ShaderManager
    Rendering.Renderer ..> TextureManager
    Rendering.Renderer ..> BufferManager
    Rendering.Renderer ..> Framebuffer
    Rendering.Renderer ..> Camera
    Rendering.Renderer ..> Primitives
    Rendering.Renderer ..> IModel : renders

    Models.SceneManager ..> IModel : manages instances
    Models.SceneManager ..> BaseSceneLoader
    Models.JsonSceneLoader --|> BaseSceneLoader
    BaseSceneLoader ..> IScene : creates
    Models.FbxLoader --|> BaseAssetLoader
    Models.ObjLoader --|> BaseAssetLoader
    BaseAssetLoader ..> IModel : creates
    
    DataLoaders.UniformedFrameLoader --|> BaseFrameLoader
    DataLoaders.SeparatedFrameLoader --|> BaseFrameLoader
    BaseFrameLoader ..> Frame : creates

    Rendering.Renderer --|> core.IRenderer
    Models.Model --|> core.IModel
    Models.SceneManager --|> core.IScene
    DataLoaders.BaseFrameLoader --|> core.IFrameLoader
    OcclusionProviders.ConcreteOcclusionProvider --|> core.IOcclusionProvider
    Trackers.ApriltagTracker --|> core.ITracker
    
    Systems.OcclusionSystem ..> Utils.TransformUtils
    Systems.VisualizeSystem ..> Utils.TransformUtils
    Rendering.Renderer ..> Utils.TransformUtils
    Models.SceneManager ..> Utils.TransformUtils
```

## 6. 実装手順 (高レベル)

1.  **コアインターフェースの定義:** `core/`ディレクトリにすべての抽象基底クラスを作成します。
2.  **ユーティリティの実装:** `Utils/TransformUtils.py`を開発し、`Utils/Logger.py`を移動/検証します。
3.  **モデルサブシステム (`Models/`) の開発:**
    *   `Model.py`, `Mesh.py`, `Material.py`, `Texture.py`を定義します。
    *   `BaseAssetLoader.py`および具体的なローダー (`FbxLoader.py`, `ObjLoader.py`) を実装します。
    *   `BaseSceneLoader.py`および`JsonSceneLoader.py`を実装します。
    *   `SceneManager.py`を開発します。
4.  **レンダリングサブシステム (`Rendering/`) の開発:**
    *   `ShaderManager.py`, `TextureManager.py`, `BufferManager.py`, `Framebuffer.py`, `Camera.py`, `Primitives.py`を実装します。
    *   `Models/`サブシステムからモデルをレンダリングでき、深度のみのレンダリングパスをサポートするように、モダンOpenGLを使用してメインの`Renderer.py`を実装します。
5.  **データローダー (`DataLoaders/`) のリファクタリング:**
    *   `Frame.py`を移動します。
    *   `BaseFrameLoader.py`を実装します（将来のローダータイプに有益と判断された場合）。
    *   既存のデータ読み込みロジックを`UniformedFrameLoader.py`にリファクタリングします。
6.  **具体的な実装のリファクタリング:**
    *   `Trackers/ApriltagTracker.py`を移動および適応させます。
    *   `OcclusionProviders/`の実装を移動および適応させます。
7.  **システム (`Systems/`) のリファクタリング:**
    *   新しいサブシステムとインターフェースを利用するように`OcclusionSystem.py`と`VisualizeSystem.py`をリファクタリングします。
    *   `OcclusionProcessor.py`を実装します。
8.  **テスト:** 正確性と安定性を確保するために、各段階で徹底的な単体テストと統合テストを実施します。

## 7. 結論

このリファクタリングは、MRオクルージョンフレームワークのためのより堅牢で、保守可能で、拡張可能な基盤を作成することを目的としています。現在のアーキテクチャの問題に対処し、最新のソフトウェア設計原則を採用することにより、フレームワークは将来の開発と機能強化に対してより良い位置付けになります。