Ext.define('ClassificationApp.view.training.TrainController', {
    extend: 'Ext.app.ViewController',
    alias: 'controller.training-train',

    refresh: function(panel, tool, event) {
        ClassificationApp.getApplication().getStore('PlannedTrainings').load({
            callback: function (r, options, success) {
                if (success === true) {
                    
                }
            }
        });

        
    },

    form_new_planned_training: function(panel, tool, event) {
        var myForm = new Ext.form.Panel({
            width: 500,
            height: 400,
            bodyPadding: 5,
            title: "Vytvoření plánovaného trénování",
            layout: 'anchor',
            defaults: {
                anchor: '100%'
            },
            floating: true,
            closable : true,
            defaultType: 'textfield',
            items: [
                {
                    fieldLabel: 'Data',
                    xtype: 'combobox',
                    store: 'TrainingData',
                    queryMode: 'local',
                    displayField: 'data',
                    valueField: 'data',
                    name: 'data',
                    allowBlank: false
                },
                {
                    fieldLabel: 'Poznámka',
                    name: 'note',
                },
                {
                    xtype: 'datefield',
                    fieldLabel: 'Dátum spuštění',
                    format: "Y-m-d",
                    name: 'date',
                    allowBlank: false
                },
                {
                    xtype: 'timefield',
                    fieldLabel: 'Čas spuštění',
                    format: "H:i",
                    name: 'time',
                    allowBlank: false
                },
                {
                    fieldLabel: 'E-mail',
                    name: 'email',
                },
            ],
            buttons: [
                { text: 'Zrušit',
                handler: function() {
                    myForm.close();
                } },
                { 
                    text: 'Vytvořit',
                    formBind: true,
                    disabled: true,
                    handler: function() {

                        var newData = myForm.getForm().findField('data').getSubmitValue();
                        var newNote = myForm.getForm().findField('note').getSubmitValue();
                        var newDate = myForm.getForm().findField('date').getSubmitValue();
                        var newTime = myForm.getForm().findField('time').getSubmitValue();
                        var newEmail = myForm.getForm().findField('email').getSubmitValue();
                        newDate = newDate + ' ' + newTime;
                        var store = panel.getStore();
                        var new_planned_training = Ext.create('ClassificationApp.store.PlannedTrainings', {
                            data: newData, note: newNote, date: newDate, email: newEmail
                        });
                        store.add(new_planned_training);
                        store.sync();
                        myForm.close();
                }}
              ]
        });
        myForm.show();
    },

    form_delete_planned_training: function(grid, rowIndex, colIndex) {
        var rec = grid.getStore().getAt(rowIndex);
        Ext.Msg.show({
            title: "Vymazání plánovaného trénovaní",
            message: "Opravdu chcete smazat plánované trénování? Pokud si přejete plánované trénování vymazat stiskněte Ano, jinak stiskněte Ne. Provedené změny nelze vrátit",
            buttons: Ext.Msg.YESNO,
            buttonText: {
                yes: 'Ano',
                no: 'Ne'
            },
            icon: Ext.Msg.QUESTION,
            fn: function(btn) {
                if (btn === 'yes') {
                    grid.getStore().remove([rowIndex]);
                    grid.getStore().sync();
                } else {
                    // console.log('Cancel pressed');
                }
            },
        });
    },

    form_edit_planned_training: function(grid, rowIndex, colIndex) {
        var plannedTraining = grid.getStore().getAt(rowIndex);
        var myForm = new Ext.form.Panel({
            width: 500,
            height: 400,
            bodyPadding: 5,
            title: "Úprava plánovaného trénovaní",
            layout: 'anchor',
            defaults: {
                anchor: '100%'
            },
            floating: true,
            closable : true,
            defaultType: 'textfield',
            items: [
                {
                    fieldLabel: 'Data',
                    xtype: 'combobox',
                    store: 'TrainingData',
                    queryMode: 'local',
                    displayField: 'data',
                    valueField: 'data',
                    name: 'data',
                    value: plannedTraining.get('data'),
                    allowBlank: false
                },
                {
                    fieldLabel: 'Poznámka',
                    name: 'note',
                    value: plannedTraining.get('note'),
                },
                {
                    xtype: 'datefield',
                    fieldLabel: 'Dátum spuštění',
                    name: 'date',
                    format: 'Y-m-d',
                    value: plannedTraining.get('date'),
                    allowBlank: false
                },
                {
                    xtype: 'timefield',
                    fieldLabel: 'Čas spuštění',
                    name: 'time',
                    value: plannedTraining.get('date'),
                    allowBlank: false
                },
                {
                    fieldLabel: 'E-mail',
                    name: 'email',
                    value: plannedTraining.get('email'),
                },
            ],
            buttons: [
                { text: 'Zrušit',
                handler: function() {
                    myForm.close();
                } },
                { 
                    text: 'Uložit',
                    formBind: true,
                    disabled: true,
                    handler: function() {

                        var values = myForm.getForm().getValues();
                        values.date = values.date + ' ' + values.time;
                        var store = grid.getStore();
                        plannedTraining.set(values);
                        store.sync();
                        myForm.close();
                }}
              ]
        });
        myForm.show();
    },
});
