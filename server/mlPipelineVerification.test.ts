import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {
  estimateShelfLife,
  evaluateEatFirstPriority,
  evaluateFreshnessPolicy,
  inferFoodForm,
  isFreshnessSupportedFood,
} from './foodServices.ts';
import { normalizeApiBaseUrl } from '../frontend/src/config/api.js';

const EVAL_DIR = path.resolve(process.cwd(), 'data', 'real_world_eval');

test('inferFoodForm handles fresh, cut, sliced, diced, dried, powdered, cooked, processed, and chilli edge cases', () => {
  assert.equal(inferFoodForm('Green Chilli', 'Fresh'), 'fresh');
  assert.equal(inferFoodForm('dried red chilli'), 'dried');
  assert.equal(inferFoodForm('red chilli'), 'dried');
  assert.equal(inferFoodForm('Chili Pepper', 'Rotten'), 'unknown');
  assert.equal(inferFoodForm('cut apple'), 'cut');
  assert.equal(inferFoodForm('sliced tomato'), 'sliced');
  assert.equal(inferFoodForm('diced mango'), 'diced');
  assert.equal(inferFoodForm('chilli powder'), 'powdered');
  assert.equal(inferFoodForm('roasted potato'), 'cooked');
  assert.equal(inferFoodForm('canned tomato'), 'processed');
  assert.equal(inferFoodForm('Bell Pepper'), 'fresh');
  assert.notEqual(inferFoodForm('green chilli'), inferFoodForm('dried red chilli'));
});

test('freshness policy keeps produce in Fresh/Slightly Spoiled/Rotten and never fabricates missing scores', () => {
  const apple = evaluateFreshnessPolicy({
    detectedFood: 'Apple',
    foodForm: 'fresh',
    rawFreshnessLabel: 'Fresh',
    rawFreshnessScore: 91.5,
  });
  assert.equal(apple.status, 'success');
  assert.equal(apple.label, 'Fresh');
  assert.equal(apple.score, 91.5);
  assert.equal(apple.confidence, 91.5);

  const rottenTomato = evaluateFreshnessPolicy({
    detectedFood: 'Tomato',
    foodForm: 'fresh',
    rawFreshnessLabel: 'Rotten',
    rawFreshnessScore: 98.2,
  });
  assert.equal(rottenTomato.status, 'success');
  assert.equal(rottenTomato.label, 'Rotten');
  assert.equal(rottenTomato.score, 98.2);

  const missingScore = evaluateFreshnessPolicy({
    detectedFood: 'Banana',
    foodForm: 'fresh',
    rawFreshnessLabel: 'Slightly Spoiled',
    rawFreshnessScore: undefined,
  });
  assert.equal(missingScore.status, 'success');
  assert.equal(missingScore.label, 'Slightly Spoiled');
  assert.equal(missingScore.score, null);
  assert.equal(missingScore.confidence, null);

  const missingLabel = evaluateFreshnessPolicy({
    detectedFood: 'Banana',
    foodForm: 'fresh',
    rawFreshnessLabel: undefined,
    rawFreshnessScore: undefined,
  });
  assert.equal(missingLabel.status, 'unavailable');
  assert.equal(missingLabel.label, 'Freshness assessment unavailable.');
  assert.equal(missingLabel.score, null);
  assert.equal(missingLabel.confidence, null);
});

test('freshness policy does NOT force bakery products or unsupported categories/forms into produce classes', () => {
  for (const bakery of ['Bread', 'Cake', 'Muffin', 'Croissant', 'Bagel', 'Toast', 'Bun']) {
    const res = evaluateFreshnessPolicy({
      detectedFood: bakery,
      foodForm: 'fresh',
      rawFreshnessLabel: 'Rotten',
      rawFreshnessScore: 99,
    });
    assert.equal(res.status, 'unavailable', `${bakery} must be unavailable`);
    assert.equal(res.label, 'Freshness assessment unavailable.');
    assert.equal(res.confidence, null);
    assert.equal(res.score, null);
  }

  for (const unsupported of ['Milk', 'Cheese', 'Yogurt', 'Chicken', 'Pizza', 'Rice', 'Coffee', 'QuantumFood']) {
    const res = evaluateFreshnessPolicy({
      detectedFood: unsupported,
      foodForm: 'fresh',
      rawFreshnessLabel: 'Fresh',
      rawFreshnessScore: 88,
    });
    assert.equal(res.status, 'unavailable', `${unsupported} must be unavailable`);
    assert.equal(res.label, 'Freshness assessment unavailable.');
    assert.equal(res.confidence, null);
    assert.equal(res.score, null);
  }

  for (const form of ['dried', 'powdered', 'cooked', 'processed']) {
    const res = evaluateFreshnessPolicy({
      detectedFood: 'Tomato',
      foodForm: form,
      rawFreshnessLabel: 'Fresh',
      rawFreshnessScore: 95,
    });
    assert.equal(res.status, 'unavailable');
    assert.equal(res.label, 'Freshness assessment unavailable.');
    assert.equal(res.confidence, null);
    assert.equal(res.score, null);
  }

  const lowConf = evaluateFreshnessPolicy({
    detectedFood: 'Apple',
    foodForm: 'fresh',
    rawFreshnessLabel: 'Fresh',
    rawFreshnessScore: 90,
    isLowConfidence: true,
  });
  assert.equal(lowConf.status, 'unavailable');
  assert.equal(lowConf.label, 'Freshness assessment unavailable.');
  assert.equal(lowConf.score, null);

  assert.equal(isFreshnessSupportedFood('Bread', 'fresh'), false);
  assert.equal(isFreshnessSupportedFood('Apple', 'fresh'), true);
  assert.equal(isFreshnessSupportedFood('Apple', 'cut'), true);
  assert.equal(isFreshnessSupportedFood('Apple', 'dried'), false);
});

test('estimateShelfLife keeps USDA FoodKeeper reference separate from remaining quality and marks non-ML-regression', () => {
  const appleCounter = estimateShelfLife('Apple', 'countertop', 0, 'Fresh', 'fresh');
  assert.equal(appleCounter.status, 'available');
  assert.equal(appleCounter.isMlRegression, false);
  assert.deepEqual(appleCounter.referenceDuration, { minDays: 5, maxDays: 7 });
  assert.deepEqual(appleCounter.remaining, { minDays: 5, maxDays: 7 });
  assert.equal(appleCounter.source, 'USDA FoodKeeper');

  const appleFridge = estimateShelfLife('Apple', 'fridge', 0, 'Fresh', 'fresh');
  assert.equal(appleFridge.status, 'available');
  assert.deepEqual(appleFridge.referenceDuration, { minDays: 15, maxDays: 15 });
  assert.deepEqual(appleFridge.remaining, { minDays: 15, maxDays: 15 });

  const cutAppleCounter = estimateShelfLife('Apple', 'countertop', 0, 'Fresh', 'cut');
  assert.equal(cutAppleCounter.status, 'available');
  assert.deepEqual(cutAppleCounter.referenceDuration, { minDays: 1, maxDays: 2 });
  assert.ok(cutAppleCounter.remaining.maxDays <= 2);

  const potatoFridge = estimateShelfLife('Potato', 'fridge', 0, 'Fresh', 'fresh');
  assert.equal(potatoFridge.status, 'unavailable');
  assert.equal(potatoFridge.isMlRegression, false);

  const potatoCounter = estimateShelfLife('Potato', 'countertop', 0, 'Fresh', 'fresh');
  assert.equal(potatoCounter.status, 'available');

  const driedChilli = estimateShelfLife('dried red chilli', 'countertop', 0, 'Rotten', 'dried');
  assert.equal(driedChilli.status, 'unavailable');
  assert.equal(driedChilli.matchType, 'form_unsupported');
  assert.equal(driedChilli.remaining, null);

  const spoiledTomato = estimateShelfLife('Tomato', 'countertop', 0, 'Slightly Spoiled', 'fresh');
  assert.equal(spoiledTomato.status, 'available');
  assert.deepEqual(spoiledTomato.referenceDuration, { minDays: 3, maxDays: 5 });
  assert.ok(spoiledTomato.remaining.maxDays < spoiledTomato.referenceDuration.maxDays);
  assert.equal(spoiledTomato.heuristicApplied, true);

  const rottenOrange = estimateShelfLife('Orange', 'countertop', 0, 'Rotten', 'fresh');
  assert.equal(rottenOrange.status, 'available');
  assert.deepEqual(rottenOrange.remaining, { minDays: 0, maxDays: 0 });
  assert.ok(rottenOrange.referenceDuration.maxDays > 0);

  const slicedBread = estimateShelfLife('Bread', 'countertop', 0, null, 'sliced');
  assert.equal(slicedBread.status, 'available');
  assert.deepEqual(slicedBread.referenceDuration, { minDays: 4, maxDays: 7 });
  assert.deepEqual(slicedBread.remaining, { minDays: 4, maxDays: 7 });
  assert.equal(slicedBread.source, 'USDA FoodKeeper');
  assert.equal(slicedBread.isMlRegression, false);

  const unknown = estimateShelfLife('QuantumSpaghetti123', 'countertop', 0, 'Fresh', 'fresh');
  assert.equal(unknown.status, 'unavailable');
  assert.equal(unknown.remaining, null);
});

test('evaluateEatFirstPriority deterministically ranks VERY_HIGH, HIGH, MEDIUM, LOW, and UNAVAILABLE', () => {
  const unavailable = evaluateEatFirstPriority({ status: 'unavailable' }, 'Fresh');
  assert.equal(unavailable.status, 'unavailable');
  assert.equal(unavailable.score, 0);
  assert.equal(unavailable.priority, null);

  const rottenSl = estimateShelfLife('Tomato', 'countertop', 0, 'Rotten', 'fresh');
  const veryHigh = evaluateEatFirstPriority(rottenSl, 'Rotten');
  assert.equal(veryHigh.priority, 'VERY_HIGH');
  assert.equal(veryHigh.score, 4);

  const spoiledSl = estimateShelfLife('Banana', 'countertop', 0, 'Slightly Spoiled', 'fresh');
  const high = evaluateEatFirstPriority(spoiledSl, 'Slightly Spoiled');
  assert.equal(high.priority, 'HIGH');
  assert.equal(high.score, 3);

  const freshAppleCounter = estimateShelfLife('Apple', 'countertop', 0, 'Fresh', 'fresh');
  const medium = evaluateEatFirstPriority(freshAppleCounter, 'Fresh');
  assert.equal(medium.priority, 'MEDIUM');
  assert.equal(medium.score, 2);

  const freshAppleFridge = estimateShelfLife('Apple', 'fridge', 0, 'Fresh', 'fresh');
  const low = evaluateEatFirstPriority(freshAppleFridge, 'Fresh');
  assert.equal(low.priority, 'LOW');
  assert.equal(low.score, 1);
});

test('real_world_eval dataset fixtures exist and have valid JPEG signatures', () => {
  const expectedFiles = [
    'apple.jpg',
    'banana.jpg',
    'bell_pepper.jpg',
    'bread.jpg',
    'green_chilli.jpg',
    'mango.jpg',
    'multi_apple_banana_tomato.jpg',
    'orange.jpg',
    'pomegranate.jpg',
    'potato.jpg',
    'rotten_orange.jpg',
    'rotten_tomato.jpg',
    'tomato.jpg',
  ];
  for (const name of expectedFiles) {
    const p = path.join(EVAL_DIR, name);
    assert.ok(fs.existsSync(p), `Missing real-world eval image: ${name}`);
    const buf = fs.readFileSync(p);
    assert.ok(buf.length > 1000, `Unexpected tiny file: ${name}`);
    assert.equal(buf[0], 0xff, `${name} JPEG SOI byte 0`);
    assert.equal(buf[1], 0xd8, `${name} JPEG SOI byte 1`);
  }
});

test('normalizeApiBaseUrl safely resolves bare port VITE_API_BASE_URL=3000 across environments', () => {
  assert.equal(
    normalizeApiBaseUrl('3000', { protocol: 'http:', hostname: 'localhost', port: '3000' }),
    '',
  );
  assert.equal(
    normalizeApiBaseUrl('3000', {
      protocol: 'https:',
      hostname: 'ais-dev-ywqyilkrsbnxowxreew4ht-911925248929.asia-east1.run.app',
      port: '',
    }),
    '',
  );
  assert.equal(
    normalizeApiBaseUrl('3000', { protocol: 'http:', hostname: 'localhost', port: '5173' }),
    'http://localhost:3000',
  );
  assert.equal(
    normalizeApiBaseUrl('http://localhost:3000/', { protocol: 'http:', hostname: 'localhost', port: '5173' }),
    'http://localhost:3000',
  );
  assert.equal(
    normalizeApiBaseUrl('http://localhost:3000/', {
      protocol: 'https:',
      hostname: 'ais-dev-ywqyilkrsbnxowxreew4ht-911925248929.asia-east1.run.app',
      port: '',
    }),
    '',
  );
  assert.equal(
    normalizeApiBaseUrl('localhost:3000', {
      protocol: 'https:',
      hostname: 'ais-dev-ywqyilkrsbnxowxreew4ht-911925248929.asia-east1.run.app',
      port: '',
    }),
    '',
  );
  assert.equal(
    normalizeApiBaseUrl('http://127.0.0.1:8000/', {
      protocol: 'https:',
      hostname: 'ais-pre-ywqyilkrsbnxowxreew4ht-911925248929.asia-east1.run.app',
      port: '',
    }),
    '',
  );
  assert.equal(
    normalizeApiBaseUrl('/api', {
      protocol: 'https:',
      hostname: 'ais-dev-ywqyilkrsbnxowxreew4ht-911925248929.asia-east1.run.app',
      port: '',
    }),
    '',
  );
  assert.equal(
    normalizeApiBaseUrl('https://ais-dev-ywqyilkrsbnxowxreew4ht-911925248929.asia-east1.run.app/api/', {
      protocol: 'https:',
      hostname: 'ais-dev-ywqyilkrsbnxowxreew4ht-911925248929.asia-east1.run.app',
      port: '',
    }),
    'https://ais-dev-ywqyilkrsbnxowxreew4ht-911925248929.asia-east1.run.app',
  );
  assert.equal(normalizeApiBaseUrl('', { protocol: 'http:', hostname: 'localhost', port: '3000' }), '');
});
