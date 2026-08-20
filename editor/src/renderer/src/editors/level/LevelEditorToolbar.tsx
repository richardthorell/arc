import { MainToolbar, type MainToolbarProps } from '../../layout/MainToolbar';

export type LevelEditorToolbarProps = MainToolbarProps;

export function LevelEditorToolbar(props: LevelEditorToolbarProps) {
  return <MainToolbar {...props} />;
}
