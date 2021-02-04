Ext.define('ClassificationApp.view.training.TrainModel', {
    extend: 'Ext.app.ViewModel',
    alias: 'viewmodel.training-train',
    data: {
        tooltip_refresh: 'Obnovit data',
        tooltip_new_planned_classification: 'Vytvořit nové plánované trénování',
        tooltip_delete: 'Vymazání plánovaného trénování',
        tooltip_edit: 'Změna plánovaného trénování', 
        table_data: 'Dáta',
        talbe_model: 'Model',
        table_note: 'Poznámka',
        table_date: 'Dátum spuštění',
        table_email: 'E-mail',
        table_status: 'Status',
        table_error: 'Chyba',
    }

});
