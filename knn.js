/**
 * k-Nearest Neighbors (k-NN) Core Algorithm
 * Includes distance metrics, Min-Max normalization, prediction engine, and cross-validation.
 */

// Distance metrics: Euclidean (L2), Manhattan (L1), Chebyshev (L-inf), Cosine
const DIST = {
  euclid: (a, b) => {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i];
      s += d * d;
    }
    return Math.sqrt(s);
  },
  manhattan: (a, b) => {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      s += Math.abs(a[i] - b[i]);
    }
    return s;
  },
  chebyshev: (a, b) => {
    let m = 0;
    for (let i = 0; i < a.length; i++) {
      m = Math.max(m, Math.abs(a[i] - b[i]));
    }
    return m;
  },
  cosine: (a, b) => {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return (na > 0 && nb > 0) ? Math.max(0, 1 - dot / (Math.sqrt(na) * Math.sqrt(nb))) : 1;
  }
};

// Compute Min-Max bounding box across dimensions
function getBounds(data = (typeof STATE !== 'undefined' ? STATE.data : [])) {
  if (!data || data.length === 0) return { min: [0, 0, 0], max: [1, 1, 1] };
  const dims = 3;
  const min = Array(dims).fill(Infinity);
  const max = Array(dims).fill(-Infinity);

  data.forEach(p => {
    const coords = [p.x, p.y, p.z || 0];
    for (let i = 0; i < dims; i++) {
      if (coords[i] < min[i]) min[i] = coords[i];
      if (coords[i] > max[i]) max[i] = coords[i];
    }
  });

  for (let i = 0; i < dims; i++) {
    if (min[i] === max[i]) { min[i] -= 1; max[i] += 1; }
  }
  return { min, max };
}

// Normalize coordinates to [0, 1] using Min-Max bounds
function normalizePoint(coords, bounds, dims = 2) {
  if (!bounds || !bounds.min || !bounds.max) return coords.slice(0, dims);
  return coords.slice(0, dims).map((v, i) => {
    const span = (bounds.max[i] - bounds.min[i]) || 1;
    return (v - bounds.min[i]) / span;
  });
}

/**
 * k-NN prediction engine for classification and regression.
 * - Distance calculation & stable sort
 * - Select top-k nearest neighbors
 * - Classification: Majority vote or inverse distance weighting (1/d), with 1-NN tie-breaking
 * - Regression: Mean or inverse distance weighting (1/d IDW)
 */
function predictKNN(pointCoords, customK, opts = {}) {
  const state = (typeof STATE !== 'undefined') ? STATE : {};
  const data = opts.data || state.data || [];
  if (!data || data.length === 0) {
    return { label: -1, nearest: [], votes: {}, counts: {}, weights: {}, r: 0, distances: [], isRegression: false };
  }

  const kVal = (customK !== undefined) ? customK : (opts.k !== undefined ? opts.k : (state.k || 5));
  const metric = opts.metric || state.metric || 'euclid';
  const isNorm = (opts.norm !== undefined) ? opts.norm : (state.norm || false);
  const isWeighted = (opts.weighting !== undefined) ? opts.weighting : (state.weighting || false);
  const task = opts.task || state.task || 'classification';

  // Infer classes from dataset if not provided
  let classes = opts.classes || state.classes;
  if (!classes || classes.length === 0) {
    const uniqueIds = Array.from(new Set(data.map(p => (p.c !== undefined ? p.c : 0)))).sort((a, b) => a - b);
    classes = uniqueIds.map(cid => ({ id: cid, name: `Class ${cid}` }));
  }

  const dims = 2;
  const fDist = DIST[metric] || DIST.euclid;
  const bounds = isNorm ? getBounds(data) : { min: null, max: null };

  const normPoint = isNorm
    ? normalizePoint(pointCoords, bounds, dims)
    : pointCoords.slice(0, dims);

  // 1. Calculate distances to all training points
  const distances = data.map((p, idx) => {
    const raw = [p.x, p.y, p.z || 0].slice(0, dims);
    const pCoords = isNorm
      ? normalizePoint(raw, bounds, dims)
      : raw;
    return {
      index: idx,
      point: p,
      dist: fDist(normPoint, pCoords)
    };
  });

  // 2. Stable sort by distance
  distances.sort((a, b) => a.dist - b.dist || a.index - b.index);

  // 3. Extract k nearest neighbors
  const kActual = Math.min(kVal, distances.length);
  const nearest = distances.slice(0, kActual);
  const r = nearest.length > 0 ? nearest[nearest.length - 1].dist : 0;

  // 4. Regression task
  if (task === 'regression') {
    let sumY = 0;
    let sumW = 0;
    let minY = Infinity, maxY = -Infinity;

    nearest.forEach(n => {
      const v = (n.point.val !== undefined) ? n.point.val : n.point.y;
      if (v < minY) minY = v;
      if (v > maxY) maxY = v;
      const w = isWeighted ? 1 / (n.dist + 1e-5) : 1;
      sumY += v * w;
      sumW += w;
    });

    const yHat = sumW > 0 ? (sumY / sumW) : 0;
    const yMean = nearest.length > 0
      ? (nearest.reduce((s, n) => s + ((n.point.val !== undefined) ? n.point.val : n.point.y), 0) / nearest.length)
      : 0;

    return {
      label: -1,
      isRegression: true,
      yHat: +yHat.toFixed(2),
      yMean: +yMean.toFixed(2),
      yMin: minY !== Infinity ? +minY.toFixed(2) : 0,
      yMax: maxY !== -Infinity ? +maxY.toFixed(2) : 0,
      nearest,
      votes: {},
      counts: {},
      weights: {},
      r,
      distances
    };
  }

  // 5. Classification task: Tally votes and weights
  const votes = {};
  const weights = {};
  classes.forEach(c => {
    votes[c.id] = 0;
    weights[c.id] = 0;
  });

  nearest.forEach(n => {
    const cid = n.point.c;
    if (votes[cid] !== undefined) votes[cid]++;
    const w = isWeighted ? 1 / (n.dist + 1e-5) : 1;
    if (weights[cid] !== undefined) weights[cid] += w;
  });

  let bestClass = -1;
  let maxScore = -1;
  const tiedClasses = [];

  classes.forEach(c => {
    const score = isWeighted ? weights[c.id] : votes[c.id];
    if (score > maxScore) {
      maxScore = score;
      bestClass = c.id;
      tiedClasses.length = 0;
      tiedClasses.push(c.id);
    } else if (score === maxScore && score > 0) {
      tiedClasses.push(c.id);
    }
  });

  // Tie-breaking: 1-NN nearest neighbor fallback
  if (tiedClasses.length > 1) {
    for (const n of nearest) {
      if (tiedClasses.includes(n.point.c)) {
        bestClass = n.point.c;
        break;
      }
    }
  }

  return {
    label: bestClass,
    nearest,
    votes,
    counts: votes,
    weights,
    r,
    distances,
    isRegression: false
  };
}

/**
 * Leave-One-Out Cross-Validation (LOOCV) evaluation.
 * Returns accuracy (%) for classification or MSE/RMSE for regression.
 */
function evaluateLOOCV(data = (typeof STATE !== 'undefined' ? STATE.data : []), opts = {}) {
  if (!data || data.length <= 1) return { accuracy: 0, mse: 0, rmse: 0, total: 0, correct: 0 };
  const kVal = opts.k !== undefined ? opts.k : (typeof STATE !== 'undefined' ? STATE.k : 5);
  const metric = opts.metric || (typeof STATE !== 'undefined' ? STATE.metric : 'euclid');
  const isNorm = opts.norm !== undefined ? opts.norm : (typeof STATE !== 'undefined' ? STATE.norm : false);
  const isWeighted = opts.weighting !== undefined ? opts.weighting : (typeof STATE !== 'undefined' ? STATE.weighting : false);
  const task = opts.task || (typeof STATE !== 'undefined' ? STATE.task : 'classification');
  const classes = opts.classes || (typeof STATE !== 'undefined' ? STATE.classes : []);

  let correct = 0;
  let totalErrorSq = 0;

  for (let i = 0; i < data.length; i++) {
    const testPoint = data[i];
    const trainData = data.filter((_, idx) => idx !== i);
    const pred = predictKNN([testPoint.x, testPoint.y], kVal, {
      data: trainData,
      metric,
      norm: isNorm,
      weighting: isWeighted,
      task,
      classes
    });

    if (task === 'regression') {
      const actualY = (testPoint.val !== undefined) ? testPoint.val : testPoint.y;
      const diff = pred.yHat - actualY;
      totalErrorSq += diff * diff;
    } else {
      if (pred.label === testPoint.c) {
        correct++;
      }
    }
  }

  const n = data.length;
  return {
    accuracy: task === 'classification' ? +(correct / n * 100).toFixed(2) : 0,
    mse: task === 'regression' ? +(totalErrorSq / n).toFixed(4) : 0,
    rmse: task === 'regression' ? +Math.sqrt(totalErrorSq / n).toFixed(4) : 0,
    total: n,
    correct
  };
}

// Module export for browser and Node.js
const KNN_CORE = {
  DIST,
  getBounds,
  normalizePoint,
  predictKNN,
  evaluateLOOCV
};

if (typeof window !== 'undefined') {
  window.DIST = DIST;
  window.getBounds = getBounds;
  window.normalizePoint = normalizePoint;
  window.predictKNN = predictKNN;
  window.evaluateLOOCV = evaluateLOOCV;
  window.KNN_CORE = KNN_CORE;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = KNN_CORE;
}
