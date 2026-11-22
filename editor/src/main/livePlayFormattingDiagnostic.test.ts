import { readFileSync } from 'node:fs';
import path from 'node:path';
import prettier from 'prettier';
import { it } from 'vitest';

// Temporary, removed before merge: capture exact CI Prettier output to
// repair format:check without needing an external formatting tool.
it('L1.4a formatting diagnostic', async () => {
  const files = [
    'src/main/livePlayAuthoringEdits.test.ts',
    'src/main/livePlayAuthoringEdits.ts',
    'src/main/livePlayControlServer.ts',
    'src/main/main.ts',
  ];
  for (const [index, file] of files.entries()) {
    const resolved = path.resolve(process.cwd(), file);
    const settings = (await prettier.resolveConfig(resolved)) ?? {};
    const formatted = await prettier.format(readFileSync(resolved, 'utf8'), { ...settings, filepath: resolved });
    const encoded = Buffer.from(formatted, 'utf8').toString('base64');
    const chunks = encoded.match(/.{1,800}/gu) ?? [];
    console.log(`ARC_FORMAT_FILE_${index}_COUNT_${chunks.length}`);
    for (const [part, chunk] of chunks.entries()) console.log(`ARC_FORMAT_FILE_${index}_PART_${part}_DATA_${chunk}`);
  }
});