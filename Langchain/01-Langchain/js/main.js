// 导入threejs
import * as THREE from "three";
// 导入轨道控制器
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
// 导入lil.gui
import { GUI } from "three/examples/jsm/libs/lil-gui.module.min.js";
// 导入hdr加载器
import { RGBELoader } from "three/examples/jsm/loaders/RGBELoader.js";
// 导入gltf加载器
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
// 导入draco解码器
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import { CSS3DRenderer,CSS3DObject } from 'three/addons/renderers/CSS3DRenderer.js';

// 创建场景
const scene = new THREE.Scene();

// 创建相机
const camera = new THREE.PerspectiveCamera(
  45, // 视角
  window.innerWidth / window.innerHeight, // 宽高比
  1, // 近平面
  10000 // 远平面
);

// 创建渲染器
const renderer = new THREE.WebGLRenderer({
  antialias: true, // 开启抗锯齿
});
renderer.shadowMap.enabled = true;
renderer.toneMapping = THREE.ReinhardToneMapping;
renderer.toneMappingExposure = 1;
renderer.setSize(window.innerWidth, window.innerHeight);
document.querySelector('#webgl').appendChild(renderer.domElement);

// css3drenderer
const css3drenderer = new CSS3DRenderer();
css3drenderer.setSize( window.innerWidth, window.innerHeight );
document.querySelector( '#css' ).appendChild( css3drenderer.domElement );


// 设置相机位置
camera.position.z = 800;
camera.position.y = 250;
camera.position.x = 300;
camera.lookAt(0, 0, 0);

// 添加世界坐标辅助器
const axesHelper = new THREE.AxesHelper(500);
scene.add(axesHelper);



// 添加轨道控制器
const controls = new OrbitControls(camera, renderer.domElement);
// 设置带阻尼的惯性
controls.enableDamping = true;

let clock = new THREE.Clock();

function animate() {
  let delta = clock.getDelta();
  controls.update();
  

  requestAnimationFrame(animate);
  // 渲染
  renderer.render(scene, camera);
  css3drenderer.render(scene, camera);
}


// 监听窗口变化
window.addEventListener("resize", () => {
  // 重置渲染器宽高比
  renderer.setSize(window.innerWidth, window.innerHeight);
  // 重置相机宽高比
  camera.aspect = window.innerWidth / window.innerHeight;
  // 更新相机投影矩阵
  camera.updateProjectionMatrix();
});

// rgbeLoader 加载hdr贴图
let rgbeLoader = new RGBELoader();
rgbeLoader.load("./texture/Alex_Hart-Nature_Lab_Bones_2k.hdr", (envMap) => {
  // 设置球形贴图
  // envMap.mapping = THREE.EquirectangularReflectionMapping;
  envMap.mapping = THREE.EquirectangularRefractionMapping;
  // 设置环境贴图
  // scene.background = envMap;
  scene.background = new THREE.Color(0xffffff);
  // 设置环境贴图
  scene.environment = envMap;
});
// rgbeLoader 加载hdr贴图
// 实例化加载器gltf
const gltfLoader = new GLTFLoader();
// 实例化加载器draco
const dracoLoader = new DRACOLoader();
// 设置draco路径
dracoLoader.setDecoderPath("./draco/");
// 设置gltf加载器draco解码器
gltfLoader.setDRACOLoader(dracoLoader);

// 加载模型
gltfLoader.load(
  // 模型路径
  "./model/hilda_regular_00.glb",
  // 加载完成回调
  (gltf) => {
    // scene.add(gltf.scene);
  }
);

// 创建GUI
const gui = new GUI();


// 创建1个半径为100的球体
const sphere = new THREE.Mesh(
  new THREE.SphereGeometry(40, 64, 64),
  new THREE.MeshStandardMaterial({
    color: 0x33ffff,
  })
)
sphere.castShadow = true
sphere.receiveShadow = true
scene.add(sphere);

const obj = new THREE.Object3D()

// 创建1个宽高为200px的div
const div = document.createElement("div");
div.style.width = "200px";
div.style.height = "200px";
div.style.backgroundColor = "rgba(0, 0, 255, 0.5)";
div.style.padding = "20px";
div.style.color = "white";
div.style.boxSizing = "border-box";
div.innerHTML = "Hello, 3D!";



// 创建1个css3object
const css3dobject = new CSS3DObject(div);
obj.add(css3dobject);

// 创建一个threejs里的平面跟css3d的div重合，进行挖洞
var material = new THREE.MeshStandardMaterial({
  opacity:0.5,
  color:new THREE.Color(0x111111),
  blending:THREE.NoBlending,
})
var geometry = new THREE.BoxGeometry(200,200,1)
var mesh = new THREE.Mesh(geometry,material)
mesh.castShadow = true
mesh.receiveShadow = true

obj.add(mesh)
scene.add(obj)

// 添加平行光
const light = new THREE.DirectionalLight(0xffffff, 100);

light.position.set(500, 200, 200);
light.castShadow = true;
scene.add(light);
light.shadow.camera.near = 1;
light.shadow.camera.far = 1000;
light.shadow.camera.left = -200;
light.shadow.camera.right = 200;
light.shadow.camera.top = 200;
light.shadow.camera.bottom = -200;




animate();
