let kalman;
let realX, realY;
let realVx, realVy;
let measurements = [];
let estimates = [];
let noiseSlider, processNoiseSlider;
let noiseValue, processNoiseValue;
let noiseLevel = 10;
let processNoise = 0.01;
let dt = 0.1; // 时间间隔

function setup() {
  createCanvas(800, 600);

  // 初始化真实状态 (x, y, vx, vy)
  realX = width / 2;
  realY = height / 2;
  realVx = random(-5, 5);
  realVy = random(-5, 5);

  // 初始化卡尔曼滤波器 (4 个状态变量，2 个测量变量)
  kalman = new KalmanFilter(4, 2, dt);

  // 添加初始测量值
  measurements.push(
    createVector(
      realX + random(-noiseLevel, noiseLevel),
      realY + random(-noiseLevel, noiseLevel)
    )
  );
  estimates.push(createVector(realX, realY));

  // 设置控制面板
  noiseSlider = document.getElementById("noiseSlider");
  processNoiseSlider = document.getElementById("processNoiseSlider");
  noiseValue = document.getElementById("noiseValue");
  processNoiseValue = document.getElementById("processNoiseValue");

  noiseSlider.addEventListener("input", function () {
    noiseLevel = parseFloat(this.value);
    noiseValue.textContent = noiseLevel.toFixed(1);
  });

  processNoiseSlider.addEventListener("input", function () {
    processNoise = parseFloat(this.value);
    processNoiseValue.textContent = processNoise.toFixed(3);
    kalman.Q = kalman.createDiagonalMatrix(4, processNoise);
  });
}

function draw() {
  background(255);

  // 更新真实状态 (模拟移动)
  realX += realVx;
  realY += realVy;

  // 添加一些随机加速度 (使运动更自然)
  realVx += random(-0.5, 0.5);
  realVy += random(-0.5, 0.5);

  // 边界检查
  if (realX < 0 || realX > width) realVx *= -1;
  if (realY < 0 || realY > height) realVy *= -1;

  // 生成带噪声的测量值
  let measX = realX + random(-noiseLevel, noiseLevel);
  let measY = realY + random(-noiseLevel, noiseLevel);
  measurements.push(createVector(measX, measY));

  // 限制测量历史长度
  if (measurements.length > 100) {
    measurements.shift();
  }

  // 卡尔曼滤波更新
  let measurement = [measX, measY];
  let estimate = kalman.update(measurement);
  estimates.push(createVector(estimate[0], estimate[1]));

  // 限制估计历史长度
  if (estimates.length > 100) {
    estimates.shift();
  }

  // 绘制轨迹
  stroke(0, 255, 0, 100); // 绿色 - 真实轨迹
  strokeWeight(2);
  noFill();
  beginShape();
  for (let i = 0; i < measurements.length; i++) {
    vertex(
      realX - realVx * (measurements.length - i) * dt,
      realY - realVy * (measurements.length - i) * dt
    );
  }
  endShape();

  stroke(255, 0, 0, 100); // 红色 - 测量轨迹
  beginShape();
  for (let m of measurements) {
    vertex(m.x, m.y);
  }
  endShape();

  stroke(0, 0, 255, 100); // 蓝色 - 滤波轨迹
  beginShape();
  for (let e of estimates) {
    vertex(e.x, e.y);
  }
  endShape();

  // 绘制当前位置
  fill(0, 255, 0);
  ellipse(realX, realY, 12, 12);

  fill(255, 0, 0);
  ellipse(measX, measY, 10, 10);

  fill(0, 0, 255);
  ellipse(estimate[0], estimate[1], 8, 8);

  // 显示说明文字
  fill(0);
  textSize(16);
  text("真实位置 (绿色)", 10, 30);
  text("测量位置 (红色)", 10, 50);
  text("滤波位置 (蓝色)", 10, 70);
}

// 卡尔曼滤波器类
class KalmanFilter {
  constructor(n, m, dt) {
    this.n = n; // 状态维度 (x, y, vx, vy)
    this.m = m; // 测量维度 (x, y)
    this.dt = dt; // 时间间隔

    // 状态向量 [x, y, vx, vy]
    this.x = Array.from({ length: n }).fill(0);

    // 状态转移矩阵
    this.F = [
      [1, 0, dt, 0],
      [0, 1, 0, dt],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ];

    // 测量矩阵
    this.H = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ];

    // 过程噪声协方差矩阵
    this.Q = this.createDiagonalMatrix(n, processNoise);

    // 测量噪声协方差矩阵
    this.R = this.createDiagonalMatrix(m, 10);

    // 状态协方差矩阵
    this.P = this.createDiagonalMatrix(n, 1000);

    // 单位矩阵
    this.I = this.createIdentityMatrix(n);
  }

  // 创建对角矩阵
  createDiagonalMatrix(size, value) {
    let mat = new Array(size);
    for (let i = 0; i < size; i++) {
      mat[i] = new Array(size).fill(0);
      mat[i][i] = value;
    }
    return mat;
  }

  // 创建单位矩阵
  createIdentityMatrix(size) {
    return this.createDiagonalMatrix(size, 1);
  }

  // 预测步骤
  predict() {
    // x = F * x
    this.x = this.multiplyMatrixVector(this.F, this.x);

    // P = F * P * F^T + Q
    let FT = this.transposeMatrix(this.F);
    let FPF = this.multiplyMatrices(this.F, this.multiplyMatrices(this.P, FT));
    this.P = this.addMatrices(FPF, this.Q);
  }

  // 更新步骤
  update(measurement) {
    // 先预测
    this.predict();

    // y = z - H * x (测量残差)
    let z = measurement;
    let Hx = this.multiplyMatrixVector(this.H, this.x);
    let y = this.subtractVectors(z, Hx);

    // S = H * P * H^T + R (残差协方差)
    let HT = this.transposeMatrix(this.H);
    let HPH = this.multiplyMatrices(this.H, this.multiplyMatrices(this.P, HT));
    let S = this.addMatrices(HPH, this.R);

    // K = P * H^T * S^(-1) (卡尔曼增益)
    let S_inv = this.invertMatrix(S);
    let K = this.multiplyMatrices(this.P, this.multiplyMatrices(HT, S_inv));

    // x = x + K * y (更新状态估计)
    let Ky = this.multiplyMatrixVector(K, y);
    this.x = this.addVectors(this.x, Ky);

    // P = (I - K * H) * P (更新协方差估计)
    let KH = this.multiplyMatrices(K, this.H);
    let I_KH = this.subtractMatrices(this.I, KH);
    this.P = this.multiplyMatrices(I_KH, this.P);

    return this.x;
  }

  // 矩阵转置
  transposeMatrix(mat) {
    let rows = mat.length;
    let cols = mat[0].length;
    let result = new Array(cols);
    for (let i = 0; i < cols; i++) {
      result[i] = new Array(rows);
      for (let j = 0; j < rows; j++) {
        result[i][j] = mat[j][i];
      }
    }
    return result;
  }

  // 矩阵乘法
  multiplyMatrices(a, b) {
    let aRows = a.length;
    let aCols = a[0].length;
    let bRows = b.length;
    let bCols = b[0].length;

    if (aCols !== bRows) {
      console.error("矩阵无法相乘: 第一个矩阵的列数不等于第二个矩阵的行数");
      return null;
    }

    let result = new Array(aRows);
    for (let i = 0; i < aRows; i++) {
      result[i] = new Array(bCols).fill(0);
      for (let j = 0; j < bCols; j++) {
        for (let k = 0; k < aCols; k++) {
          result[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    return result;
  }

  // 矩阵与向量相乘
  multiplyMatrixVector(mat, vec) {
    let rows = mat.length;
    let cols = mat[0].length;

    if (cols !== vec.length) {
      console.error("矩阵与向量无法相乘: 矩阵的列数不等于向量的长度");
      return null;
    }

    let result = new Array(rows).fill(0);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[i] += mat[i][j] * vec[j];
      }
    }
    return result;
  }

  // 矩阵加法
  addMatrices(a, b) {
    let rows = a.length;
    let cols = a[0].length;

    if (rows !== b.length || cols !== b[0].length) {
      console.error("矩阵无法相加: 矩阵维度不匹配");
      return null;
    }

    let result = new Array(rows);
    for (let i = 0; i < rows; i++) {
      result[i] = new Array(cols);
      for (let j = 0; j < cols; j++) {
        result[i][j] = a[i][j] + b[i][j];
      }
    }
    return result;
  }

  // 矩阵减法
  subtractMatrices(a, b) {
    let rows = a.length;
    let cols = a[0].length;

    if (rows !== b.length || cols !== b[0].length) {
      console.error("矩阵无法相减: 矩阵维度不匹配");
      return null;
    }

    let result = new Array(rows);
    for (let i = 0; i < rows; i++) {
      result[i] = new Array(cols);
      for (let j = 0; j < cols; j++) {
        result[i][j] = a[i][j] - b[i][j];
      }
    }
    return result;
  }

  // 向量加法
  addVectors(a, b) {
    if (a.length !== b.length) {
      console.error("向量无法相加: 向量长度不匹配");
      return null;
    }

    let result = new Array(a.length);
    for (let i = 0; i < a.length; i++) {
      result[i] = a[i] + b[i];
    }
    return result;
  }

  // 向量减法
  subtractVectors(a, b) {
    if (a.length !== b.length) {
      console.error("向量无法相减: 向量长度不匹配");
      return null;
    }

    let result = new Array(a.length);
    for (let i = 0; i < a.length; i++) {
      result[i] = a[i] - b[i];
    }
    return result;
  }

  // 矩阵求逆 (仅支持 2x2 矩阵)
  invertMatrix(mat) {
    let n = mat.length;

    // 对于我们的应用，主要需要 2x2 矩阵的逆
    if (n === 2) {
      let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
      if (Math.abs(det) < 1e-10) {
        console.error("矩阵不可逆: 行列式接近零");
        return this.createIdentityMatrix(n); // 返回单位矩阵作为替代
      }

      let invDet = 1 / det;
      return [
        [mat[1][1] * invDet, -mat[0][1] * invDet],
        [-mat[1][0] * invDet, mat[0][0] * invDet],
      ];
    }

    console.error("不支持的矩阵维度求逆: " + n + "x" + n);
    return null;
  }
}
